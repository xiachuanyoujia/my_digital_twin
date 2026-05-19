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
from app.models.pose import Landmark, PoseFrame

MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "pose_landmarker.task"


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

    def process_frame(self, frame: np.ndarray) -> PoseFrame | None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(mp_image)

        if not result.pose_landmarks:
            return None

        # result.pose_landmarks is list[list[NormalizedLandmark]] — one list per detected pose
        landmarks = [
            Landmark(x=lm.x, y=lm.y, z=lm.z, visibility=lm.visibility)
            for lm in result.pose_landmarks[0]
        ]

        world_landmarks = None
        if result.pose_world_landmarks:
            world_landmarks = [
                Landmark(x=lm.x, y=lm.y, z=lm.z, visibility=lm.visibility)
                for lm in result.pose_world_landmarks[0]
            ]

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
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.camera_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.camera_height)
        self.cap.set(cv2.CAP_PROP_FPS, settings.camera_fps)

    def read(self) -> np.ndarray | None:
        ret, frame = self.cap.read()
        return frame if ret else None

    def close(self) -> None:
        self.cap.release()


class PosePipeline:
    def __init__(self) -> None:
        self.camera = CameraCapture()
        self.detector = PoseDetector()

    async def stream(self) -> AsyncGenerator[PoseFrame, None]:
        import asyncio

        loop = asyncio.get_event_loop()
        while True:
            frame = await loop.run_in_executor(None, self.camera.read)
            if frame is None:
                continue
            pose_frame = await loop.run_in_executor(
                None, self.detector.process_frame, frame
            )
            if pose_frame is not None:
                yield pose_frame

    def close(self) -> None:
        self.camera.close()
        self.detector.close()
