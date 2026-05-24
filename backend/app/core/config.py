from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8000

    camera_index: int = 0
    camera_width: int = 1280
    camera_height: int = 720
    camera_fps: int = 60

    pose_model_complexity: int = 1
    pose_min_detection_confidence: float = 0.5
    pose_min_tracking_confidence: float = 0.5

    model_dir: str = "../assets/models"

    # YOLO config
    pose_backend: str = "hybrid"  # "mediapipe" | "yolo" | "hybrid"
    yolo_person_model: str = "yolov8n.pt"
    yolo_pose_model: str = "yolov8n-pose.pt"
    yolo_confidence: float = 0.5
    yolo_iou: float = 0.45

    # Calibration
    calibration_window: int = 200
    calibration_convergence: float = 0.01
    calibration_min_samples: int = 100

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
