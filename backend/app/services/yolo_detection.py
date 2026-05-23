from __future__ import annotations

import time
from collections.abc import AsyncGenerator
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

from app.core.config import settings
from app.models.pose import COCO_TO_BASIC, Landmark, PoseFrame
from app.services.pose_detection import CameraCapture, PoseDetector, PoseSmoother

MODEL_DIR = Path(__file__).parent.parent.parent / "models"


def _get_device() -> str:
    """自动检测可用设备: cuda > directml > cpu"""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda:0"
    except ImportError:
        pass
    return "cpu"


class YoloPersonDetector:
    """YOLOv8n 人物检测器 —— 返回画面中面积最大的人物边界框"""

    def __init__(self) -> None:
        self._model: YOLO | None = None
        self._device = _get_device()
        self._available: bool | None = None  # None=未初始化, True=可用, False=不可用
        print(f"[YoloPersonDetector] 初始化, device={self._device}", flush=True)

    def _load_model(self) -> YOLO | None:
        model_path = MODEL_DIR / settings.yolo_person_model
        print(f"[YoloPersonDetector] 加载模型: {model_path} (exists={model_path.exists()})", flush=True)
        if model_path.exists():
            return YOLO(str(model_path))
        try:
            model = YOLO(settings.yolo_person_model)
            return model
        except Exception as e:
            print(f"[YoloPersonDetector] 模型加载失败: {e}")
            print(f"[YoloPersonDetector] 请手动下载 {settings.yolo_person_model} 到 {MODEL_DIR}")
            return None

    @property
    def model(self) -> YOLO | None:
        if self._available is None:
            print(f"[YoloPersonDetector] 首次加载模型...", flush=True)
            self._model = self._load_model()
            self._available = self._model is not None
            if self._model is not None:
                print(f"[YoloPersonDetector] 模型加载成功, 移至设备: {self._device}", flush=True)
                self._model.to(self._device)
            else:
                print(f"[YoloPersonDetector] 模型加载失败! 人物检测将不可用", flush=True)
        return self._model

    def detect(self, frame: np.ndarray) -> tuple[int, int, int, int] | None:
        """检测画面中的人物，返回最大人物的边界框 (x1, y1, x2, y2)，无人则返回 None"""
        if self.model is None:
            return None
        results = self.model(frame, conf=settings.yolo_confidence, iou=settings.yolo_iou,
                             classes=[0], verbose=False)  # class 0 = person

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return None

        # 选面积最大的人物
        best_area = 0
        best_bbox = None
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                best_area = area
                best_bbox = (x1, y1, x2, y2)

        self._detect_count = getattr(self, '_detect_count', 0) + 1
        if self._detect_count <= 3 or self._detect_count % 30 == 0:
            print(f"[YoloPersonDetector] detect #{self._detect_count}: bbox={best_bbox}", flush=True)

        return best_bbox

    def detect_with_confidence(self, frame: np.ndarray) -> tuple[int, int, int, int, float] | None:
        """检测人物并返回 (x1, y1, x2, y2, confidence)，无人则返回 None"""
        if self.model is None:
            return None
        results = self.model(frame, conf=settings.yolo_confidence, iou=settings.yolo_iou,
                             classes=[0], verbose=False)

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return None

        best_area = 0
        best_result = None
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                best_area = area
                best_result = (x1, y1, x2, y2, conf)

        return best_result

    def close(self) -> None:
        self._model = None


class YoloPoseDetector:
    """YOLOv8n-pose 姿态检测 —— 直接端到端输出关键点，作为 MediaPipe 的轻量替代"""

    def __init__(self) -> None:
        self._model: YOLO | None = None
        self._device = _get_device()
        self._smoother = PoseSmoother()
        self._available: bool | None = None
        print(f"[YoloPoseDetector] 初始化, device={self._device}", flush=True)

    def _load_model(self) -> YOLO | None:
        custom_path = MODEL_DIR / "yolo_pose_custom.pt"
        if custom_path.exists():
            print(f"[YoloPoseDetector] 加载自定义模型: {custom_path}", flush=True)
            return YOLO(str(custom_path))
        model_path = MODEL_DIR / settings.yolo_pose_model
        print(f"[YoloPoseDetector] 加载模型: {model_path} (exists={model_path.exists()})", flush=True)
        if model_path.exists():
            return YOLO(str(model_path))
        try:
            return YOLO(settings.yolo_pose_model)
        except Exception as e:
            print(f"[YoloPoseDetector] 模型加载失败: {e}")
            print(f"[YoloPoseDetector] 请手动下载 {settings.yolo_pose_model} 到 {MODEL_DIR}")
            return None

    @property
    def model(self) -> YOLO | None:
        if self._available is None:
            print(f"[YoloPoseDetector] 首次加载模型...", flush=True)
            self._model = self._load_model()
            self._available = self._model is not None
            if self._model is not None:
                print(f"[YoloPoseDetector] 模型加载成功, 移至设备: {self._device}", flush=True)
                self._model.to(self._device)
            else:
                print(f"[YoloPoseDetector] 模型加载失败! 姿态检测将不可用", flush=True)
        return self._model

    def process_frame(self, frame: np.ndarray) -> PoseFrame | None:
        """YOLO 姿态检测，返回 PoseFrame"""
        if self.model is None:
            return None
        results = self.model(frame, conf=settings.yolo_confidence, iou=settings.yolo_iou,
                             classes=[0], verbose=False)

        keypoints = results[0].keypoints
        if keypoints is None or keypoints.data is None or len(keypoints.data) == 0:
            self._smoother._prev = None
            return None

        # 取第一个（置信度最高的）人物
        kpts = keypoints.data[0].cpu().numpy()  # shape: (17, 3)
        h, w = frame.shape[:2]

        # 映射 COCO 17 关键点到项目 13 关键点
        landmarks_raw = []
        for coco_idx in sorted(COCO_TO_BASIC.keys()):
            x, y, conf = kpts[coco_idx]
            landmarks_raw.append(Landmark(
                x=float(x / w),
                y=float(y / h),
                z=0.0,
                visibility=float(conf),
            ))

        landmarks = self._smoother.smooth(landmarks_raw)

        return PoseFrame(
            timestamp=time.time(),
            landmarks=landmarks,
            world_landmarks=None,
            backend="yolo",
        )

    def close(self) -> None:
        self._model = None
        self._smoother._prev = None


class HybridPosePipeline:
    """
    混合流水线: YOLOv8n 人物检测 → 裁剪 ROI → MediaPipe 姿态估计

    比纯 MediaPipe 更快更准，因为:
      1. YOLO 确定人物位置后，只对 ROI 做姿态估计（输入更小 = 更快）
      2. 减少背景噪声干扰（提升精度）
      3. 检测失败时自动降级为全帧 MediaPipe
    """

    def __init__(self) -> None:
        print("[HybridPosePipeline] 初始化...", flush=True)
        self.camera = CameraCapture()
        self.person_detector = YoloPersonDetector()
        self.mediapipe_detector = PoseDetector()
        self._roi_smoother = PoseSmoother()
        self._roi_world_smoother = PoseSmoother()
        self._running = False
        self._target_fps = 30.0
        self._min_interval = 1.0 / self._target_fps
        print("[HybridPosePipeline] 初始化完成", flush=True)

    def _crop_frame(self, frame: np.ndarray, bbox: tuple[int, int, int, int],
                    margin: float = 0.15) -> tuple[np.ndarray, float, float]:
        """裁剪并填充人物区域，返回 (cropped_frame, offset_x, offset_y)"""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox

        # 加边距
        bw, bh = x2 - x1, y2 - y1
        mx = int(bw * margin)
        my = int(bh * margin)
        x1 = max(0, x1 - mx)
        y1 = max(0, y1 - my)
        x2 = min(w, x2 + mx)
        y2 = min(h, y2 + my)

        return frame[y1:y2, x1:x2], float(x1), float(y1)

    def _remap_landmarks(self, landmarks: list[Landmark],
                         crop_w: int, crop_h: int,
                         offset_x: float, offset_y: float,
                         full_w: int, full_h: int) -> list[Landmark]:
        """将裁剪区域的归一化关键点映射回全帧坐标系"""
        result = []
        for lm in landmarks:
            # 裁剪区域中的像素坐标
            px = lm.x * crop_w + offset_x
            py = lm.y * crop_h + offset_y
            result.append(Landmark(
                x=px / full_w,
                y=py / full_h,
                z=lm.z,
                visibility=lm.visibility,
            ))
        return result

    def _process_frame_with_roi(self, frame: np.ndarray,
                                bbox: tuple[int, int, int, int]) -> PoseFrame | None:
        """使用 ROI 执行 MediaPipe 姿态估计"""
        h, w = frame.shape[:2]
        cropped, off_x, off_y = self._crop_frame(frame, bbox)
        ch, cw = cropped.shape[:2]

        if ch < 32 or cw < 32:
            return None

        # 对裁剪区域运行 MediaPipe
        rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.mediapipe_detector.landmarker.detect(mp_image)

        if not result.pose_landmarks:
            return None

        from app.models.pose import BASIC_LANDMARK_INDICES

        full = result.pose_landmarks[0]
        landmarks_cropped = [
            Landmark(x=full[i].x, y=full[i].y, z=full[i].z, visibility=full[i].visibility)
            for i in BASIC_LANDMARK_INDICES
        ]

        # 映射回全帧坐标
        landmarks_full = self._remap_landmarks(landmarks_cropped, cw, ch, off_x, off_y, w, h)
        landmarks = self._roi_smoother.smooth(landmarks_full)

        world_landmarks = None
        if result.pose_world_landmarks:
            full_world = result.pose_world_landmarks[0]
            world_cropped = [
                Landmark(x=full_world[i].x, y=full_world[i].y, z=full_world[i].z,
                         visibility=full_world[i].visibility)
                for i in BASIC_LANDMARK_INDICES
            ]
            world_landmarks = self._roi_world_smoother.smooth(world_cropped)

        return PoseFrame(
            timestamp=time.time(),
            landmarks=landmarks,
            world_landmarks=world_landmarks,
            backend="hybrid",
        )

    async def stream(self) -> AsyncGenerator[PoseFrame, None]:
        import asyncio
        import sys
        import traceback

        self._running = True
        loop = asyncio.get_event_loop()
        last_send = 0.0
        frame_count = 0
        yield_count = 0
        detect_count = 0
        none_frame_count = 0
        last_report = loop.time()

        print("[HybridPipeline] stream() 已启动, 等待帧...", flush=True)
        sys.stdout.flush()

        try:
            while self._running:
                try:
                    frame = await loop.run_in_executor(None, self.camera.read)
                except Exception as e:
                    print(f"[HybridPipeline] 摄像头读取异常: {e}", flush=True)
                    await asyncio.sleep(0.1)
                    continue

                if frame is None:
                    none_frame_count += 1
                    if none_frame_count <= 3 or none_frame_count % 30 == 0:
                        print(f"[HybridPipeline] 摄像头返回 None (第{none_frame_count}次)", flush=True)
                    await asyncio.sleep(0.01)
                    continue

                frame_count += 1
                now = loop.time()
                if now - last_report > 3.0:
                    print(f"[HybridPipeline] 状态: {frame_count}帧读取, {detect_count}次检测到人体, "
                          f"{yield_count}帧已发送, fps≈{frame_count/(now-last_report+0.001):.1f}", flush=True)
                    frame_count = 0
                    detect_count = 0
                    yield_count = 0
                    last_report = now

                # 尝试 YOLO 人物检测
                try:
                    bbox = await loop.run_in_executor(None, self.person_detector.detect, frame)
                except Exception as e:
                    print(f"[HybridPipeline] YOLO检测异常: {e}", flush=True)
                    traceback.print_exc()
                    bbox = None

                if bbox is not None:
                    detect_count += 1
                    try:
                        pose_frame = await loop.run_in_executor(
                            None, self._process_frame_with_roi, frame, bbox
                        )
                    except Exception as e:
                        print(f"[HybridPipeline] ROI处理异常: {e}", flush=True)
                        traceback.print_exc()
                        pose_frame = None
                else:
                    # 降级: 全帧 MediaPipe
                    try:
                        pose_frame = await loop.run_in_executor(
                            None, self.mediapipe_detector.process_frame, frame
                        )
                    except Exception as e:
                        print(f"[HybridPipeline] MediaPipe处理异常: {e}", flush=True)
                        traceback.print_exc()
                        pose_frame = None

                if pose_frame is not None:
                    yield_count += 1
                    now = loop.time()
                    elapsed = now - last_send
                    if elapsed < self._min_interval:
                        await asyncio.sleep(self._min_interval - elapsed)
                    last_send = loop.time()
                    yield pose_frame
        except Exception as e:
            print(f"[HybridPipeline] stream异常: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
        finally:
            self._running = False
            print(f"[HybridPipeline] stream() 已结束, 共读取{frame_count}帧, 发送{yield_count}帧", flush=True)

    def close(self) -> None:
        self._running = False
        self.camera.close()
        self.person_detector.close()
        self.mediapipe_detector.close()


def create_pipeline(backend: str) -> AsyncGenerator[PoseFrame, None] | None:
    """工厂函数: 根据后端类型创建对应的流水线"""
    from app.services.pose_detection import PosePipeline

    if backend == "mediapipe":
        return PosePipeline()
    elif backend == "yolo":
        return YoloPosePipeline()
    elif backend == "hybrid":
        return HybridPosePipeline()
    else:
        raise ValueError(f"Unknown pose backend: {backend}")


class YoloPosePipeline:
    """纯 YOLO-pose 流水线: 摄像头 → YOLOv8n-pose → WebSocket"""

    def __init__(self) -> None:
        self.camera = CameraCapture()
        self.detector = YoloPoseDetector()
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

        print("[YoloPipeline] stream() 已启动", flush=True)
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
                    print(f"[YoloPipeline] 状态: {frame_count}帧读取, {detect_count}帧检测到人体", flush=True)
                    frame_count = 0
                    detect_count = 0
                    last_report = now

                pose_frame = await loop.run_in_executor(
                    None, self.detector.process_frame, frame
                )
                if pose_frame is not None:
                    detect_count += 1
                    now = loop.time()
                    elapsed = now - last_send
                    if elapsed < self._min_interval:
                        await asyncio.sleep(self._min_interval - elapsed)
                    last_send = loop.time()
                    yield pose_frame
        finally:
            self._running = False
            print("[YoloPipeline] stream() 已结束", flush=True)

    def close(self) -> None:
        self._running = False
        self.camera.close()
        self.detector.close()
