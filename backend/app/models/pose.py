from __future__ import annotations

from pydantic import BaseModel


class Landmark(BaseModel):
    """单个关键点的 3D 坐标 + 可见度"""

    x: float
    y: float
    z: float
    visibility: float = 1.0


class PoseFrame(BaseModel):
    """一帧完整的姿态数据"""

    timestamp: float
    landmarks: list[Landmark]  # 33 个 MediaPipe Pose 关键点
    world_landmarks: list[Landmark] | None = None  # 世界坐标系下的关键点


class CameraConfig(BaseModel):
    """摄像头配置"""

    index: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 60


class TrackingStatus(BaseModel):
    """追踪状态"""

    active: bool
    fps: float = 0.0
    latency_ms: float = 0.0
    detected: bool = False
