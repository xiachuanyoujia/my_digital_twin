from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


# MediaPipe Pose landmark indices kept for basic body skeleton (13 keypoints):
#   nose(0), L-shoulder(11), R-shoulder(12),
#   L-elbow(13), R-elbow(14), L-wrist(15), R-wrist(16),
#   L-hip(23), R-hip(24), L-knee(25), R-knee(26), L-ankle(27), R-ankle(28)
BASIC_LANDMARK_INDICES = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

# COCO 17 keypoint indices mapped to project's 13 keypoints
#   COCO: nose(0), left_eye(1), right_eye(2), left_ear(3), right_ear(4),
#         left_shoulder(5), right_shoulder(6), left_elbow(7), right_elbow(8),
#         left_wrist(9), right_wrist(10), left_hip(11), right_hip(12),
#         left_knee(13), right_knee(14), left_ankle(15), right_ankle(16)
# Project uses: nose(0), L-sh(11), R-sh(12), L-elbow(13), R-elbow(14),
#               L-wrist(15), R-wrist(16), L-hip(23), R-hip(24),
#               L-knee(25), R-knee(26), L-ankle(27), R-ankle(28)
# COCO → project index mapping (COCO index → project landmark index):
COCO_TO_BASIC = {
    0: 0,   # nose
    5: 1,   # left_shoulder → L_SHOULDER
    6: 2,   # right_shoulder → R_SHOULDER
    7: 3,   # left_elbow → L_ELBOW
    8: 4,   # right_elbow → R_ELBOW
    9: 5,   # left_wrist → L_WRIST
    10: 6,  # right_wrist → R_WRIST
    11: 7,  # left_hip → L_HIP
    12: 8,  # right_hip → R_HIP
    13: 9,  # left_knee → L_KNEE
    14: 10, # right_knee → R_KNEE
    15: 11, # left_ankle → L_ANKLE
    16: 12, # right_ankle → R_ANKLE
}


class BackendType(str, Enum):
    mediapipe = "mediapipe"
    yolo = "yolo"
    hybrid = "hybrid"


class Landmark(BaseModel):
    """单个关键点的 3D 坐标 + 可见度"""

    x: float
    y: float
    z: float
    visibility: float = 1.0


class PoseFrame(BaseModel):
    """一帧完整的姿态数据"""

    timestamp: float
    landmarks: list[Landmark]  # 13 个基础人体关键点
    world_landmarks: list[Landmark] | None = None  # 世界坐标系下的关键点
    backend: str = "mediapipe"  # 当前使用的检测后端


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
    backend: str = "mediapipe"
