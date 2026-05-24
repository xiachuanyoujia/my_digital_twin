from __future__ import annotations

import time
from collections.abc import AsyncGenerator
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options as bo
from mediapipe.tasks.python.vision import RunningMode

from app.core.config import settings
from app.core.logger import log_pose_frame, log_status
from app.models.pose import BASIC_LANDMARK_INDICES, Landmark, PoseFrame

MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "pose_landmarker_lite.task"


class PoseSmoother:
    """自适应速度低通滤波器 —— 静止时重度平滑降噪，运动时减少平滑降低延迟。

    核心思路: 基于帧间位移估算速度 velocity，动态调节混合系数 alpha。
      - velocity ≈ 0 (静止): alpha = min_alpha (重度平滑)
      - velocity 大 (运动):  alpha 逼近 max_alpha (快速响应)
    """

    def __init__(self, min_alpha: float = 0.45, max_alpha: float = 0.88) -> None:
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self._prev: list[Landmark] | None = None
        self._gap_frames = 0

    def smooth(self, landmarks: list[Landmark]) -> list[Landmark]:
        # Reset after gap in detection — clears stale motion trail immediately
        if self._gap_frames > 3:
            self._prev = None
        self._gap_frames = 0

        if self._prev is None or len(self._prev) != len(landmarks):
            self._prev = landmarks
            return landmarks

        result: list[Landmark] = []
        for curr, prev in zip(landmarks, self._prev):
            dx = curr.x - prev.x
            dy = curr.y - prev.y
            velocity = (dx * dx + dy * dy) ** 0.5

            # 速度越快 alpha 越高 (更相信当前帧), 反之 alpha 越低 (更多滤波)
            # Increased sensitivity: velocity * 120 for faster response
            t = min(1.0, velocity * 120.0)
            alpha = self.min_alpha + (self.max_alpha - self.min_alpha) * t
            alpha *= curr.visibility  # 可见度低时更依赖历史值

            result.append(Landmark(
                x=alpha * curr.x + (1 - alpha) * prev.x,
                y=alpha * curr.y + (1 - alpha) * prev.y,
                z=alpha * curr.z + (1 - alpha) * prev.z,
                visibility=curr.visibility,
            ))

        self._prev = result
        return result

    def mark_gap(self) -> None:
        """Call when no detection in a frame — count gaps to trigger reset."""
        self._gap_frames += 1


class PoseDetector:
    """MediaPipe PoseLandmarker (Task API, IMAGE mode)."""

    def __init__(self) -> None:
        base = bo.BaseOptions(model_asset_path=str(MODEL_PATH))
        options = vision.PoseLandmarkerOptions(
            base_options=base,
            running_mode=RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=settings.pose_min_detection_confidence,
            min_tracking_confidence=settings.pose_min_tracking_confidence,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)
        self._smoother = PoseSmoother()
        self._world_smoother = PoseSmoother()

    def process_frame(self, frame: np.ndarray) -> PoseFrame | None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(mp_image)

        if not result.pose_landmarks:
            self._smoother.mark_gap()
            self._world_smoother.mark_gap()
            self._no_detect_count = getattr(self, '_no_detect_count', 0) + 1
            if self._no_detect_count <= 3 or self._no_detect_count % 30 == 0:
                print(f"[PoseDetector] 未检测到人体 (第{self._no_detect_count}次) frame_shape={frame.shape}", flush=True)
            return None

        self._detect_count = getattr(self, '_detect_count', 0) + 1
        if self._detect_count <= 3 or self._detect_count % 30 == 0:
            print(f"[PoseDetector] 检测到人体 #{self._detect_count} frame_shape={frame.shape}", flush=True)

        # result.pose_landmarks is list[list[NormalizedLandmark]] — one list per detected pose
        full = result.pose_landmarks[0]
        landmarks = self._smoother.smooth([
            Landmark(x=full[i].x, y=full[i].y, z=full[i].z, visibility=full[i].visibility)
            for i in BASIC_LANDMARK_INDICES
        ])

        world_landmarks = None
        if result.pose_world_landmarks:
            full_world = result.pose_world_landmarks[0]
            world_landmarks = self._world_smoother.smooth([
                Landmark(x=full_world[i].x, y=full_world[i].y, z=full_world[i].z, visibility=full_world[i].visibility)
                for i in BASIC_LANDMARK_INDICES
            ])

        return PoseFrame(
            timestamp=time.time(),
            landmarks=landmarks,
            world_landmarks=world_landmarks,
        )

    def close(self) -> None:
        self.landmarker.close()


class CameraCapture:
    def __init__(self, camera_index: int | None = None) -> None:
        idx = camera_index if camera_index is not None else settings.camera_index
        self.cap = cv2.VideoCapture(idx)
        if not self.cap.isOpened():
            print(f"[CameraCapture] 无法打开摄像头 index={idx}, 尝试 index=0")
            self.cap = cv2.VideoCapture(0)
        actual_w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"[CameraCapture] 摄像头已打开 index={idx} "
              f"分辨率={actual_w:.0f}x{actual_h:.0f} fps={actual_fps:.0f}")

    def read(self) -> np.ndarray | None:
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def close(self) -> None:
        self.cap.release()


class PosePipeline:
    def __init__(self) -> None:
        self.camera = CameraCapture()
        self.detector = PoseDetector()
        self._running = False
        self._target_fps = 30.0
        self._min_interval = 1.0 / self._target_fps

    async def stream(self) -> AsyncGenerator[PoseFrame, None]:
        import asyncio
        import sys

        self._running = True
        loop = asyncio.get_event_loop()
        last_send = 0.0
        frame_count = 0
        detect_count = 0
        last_report = loop.time()

        msg = "[PosePipeline] 流已启动, 等待摄像头数据..."
        print(msg, flush=True)
        sys.stdout.flush()

        try:
            while self._running:
                frame = await loop.run_in_executor(None, self.camera.read)
                if frame is None:
                    await asyncio.sleep(0.01)
                    continue

                frame_count += 1
                now = loop.time()
                if now - last_report > 3.0:
                    print(f"[PosePipeline] 状态: {frame_count} 帧已读取, {detect_count} 次检测到人体, "
                          f"fps≈{frame_count/(now-last_report+0.001):.1f}", flush=True)
                    frame_count = 0
                    detect_count = 0
                    last_report = now

                t0 = loop.time()
                pose_frame = await loop.run_in_executor(
                    None, self.detector.process_frame, frame
                )
                t1 = loop.time()
                if pose_frame is not None:
                    detect_count += 1
                    log_pose_frame("MediaPipe", pose_frame.landmarks,
                                   extra=lambda t0=t0, t1=t1: f"latency={((t1-t0)*1000):.1f}ms")
                    now = loop.time()
                    elapsed = now - last_send
                    if elapsed < self._min_interval:
                        await asyncio.sleep(self._min_interval - elapsed)
                    last_send = loop.time()
                    yield pose_frame
        finally:
            self._running = False
            log_status("MediaPipe", status="stopped", frames=frame_count, detected=detect_count)

    def close(self) -> None:
        self._running = False
        self.camera.close()
        self.detector.close()
