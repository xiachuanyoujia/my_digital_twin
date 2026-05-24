from __future__ import annotations

import time
from collections.abc import Callable

# ---------------------------------------------------------------------------
# Structured logger for pose debugging
# ---------------------------------------------------------------------------

_pose_log_enabled: bool = True
_pose_log_interval: int = 30  # log every N frames
_frame_counters: dict[str, int] = {}
_start_times: dict[str, float] = {}

KEYPOINT_NAMES = [
    "nose", "L-sh", "R-sh", "L-elb", "R-elb",
    "L-wr", "R-wr", "L-hip", "R-hip", "L-knee", "R-knee",
    "L-ank", "R-ank",
]


def set_pose_log(enabled: bool, interval: int = 30) -> None:
    global _pose_log_enabled, _pose_log_interval
    _pose_log_enabled = enabled
    _pose_log_interval = interval


def _should_log(tag: str) -> bool:
    if not _pose_log_enabled:
        return False
    count = _frame_counters.get(tag, 0)
    return count <= 3 or count % _pose_log_interval == 0


def log_pose_frame(
    tag: str,
    landmarks: list,
    *,
    extra: Callable[[], str] | None = None,
) -> None:
    """Log a pose frame with standardized format.

    Args:
        tag: Source identifier (e.g. "MediaPipe", "YOLO", "Hybrid")
        landmarks: List of 13 Landmark objects
        extra: Optional callable returning extra info string
    """
    if not _should_log(tag):
        _frame_counters[tag] = _frame_counters.get(tag, 0) + 1
        return

    count = _frame_counters.get(tag, 0) + 1
    _frame_counters[tag] = count

    ts = time.time()
    if tag not in _start_times:
        _start_times[tag] = ts
    elapsed = ts - _start_times[tag]

    parts = [f"[{tag}] #{count} +{elapsed:.1f}s"]

    if landmarks and len(landmarks) >= 13:
        for i, name in enumerate(KEYPOINT_NAMES):
            lm = landmarks[i]
            parts.append(
                f"{name}=({lm.x:.4f},{lm.y:.4f},{lm.z:.4f},v{lm.visibility:.2f})"
            )
    else:
        parts.append("NO_DETECTION")

    if extra:
        parts.append(extra())

    print(" ".join(parts), flush=True)


def log_status(tag: str, **kwargs: object) -> None:
    """Log pipeline status with key=value pairs."""
    parts = [f"[{tag}.STATUS]"]
    for k, v in kwargs.items():
        parts.append(f"{k}={v}")
    print(" ".join(parts), flush=True)
