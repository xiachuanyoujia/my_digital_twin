"""Calibration API — measures error between frontend 3D model joint positions
and backend-detected landmarks, then optimizes mpToScene() scale parameters."""

from __future__ import annotations

import time

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/calibrate", tags=["calibration"])

# ---------------------------------------------------------------------------
# Module-level calibration state
# ---------------------------------------------------------------------------
_params = {"scale_x": 2.5, "scale_y": 3.5, "scale_z": 2.5}

# Running per-joint error stats  {joint_name: {"sum_sq": float, "sum": float, "n": int}}
_stats: dict[str, dict] = {}
_total_samples = 0

# Accumulated (landmark_norm, observed_3d) pairs for least-squares optimization
_opt_samples: list[dict] = []
_opt_window = 200
_frames_since_opt = 0

# Most recently sent landmarks (set by pose.py after each WS send)
_last_landmarks: list[dict] | None = None
_last_landmarks_ts: float = 0.0

# Joint name → landmark index (BASIC_LANDMARK_INDICES order)
JOINT_TO_IDX = {
    "nose": 0,
    "left_shoulder": 1, "right_shoulder": 2,
    "left_elbow": 3,    "right_elbow": 4,
    "left_wrist": 5,    "right_wrist": 6,
    "left_hip": 7,      "right_hip": 8,
    "left_knee": 9,     "right_knee": 10,
    "left_ankle": 11,   "right_ankle": 12,
}
IDX_TO_JOINT = {v: k for k, v in JOINT_TO_IDX.items()}

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class JointPosition(BaseModel):
    name: str
    x: float
    y: float
    z: float

class FrameReport(BaseModel):
    timestamp: float
    joints: list[JointPosition]

class JointError(BaseModel):
    rmse: float
    mean_error: float
    sample_count: int

class CalibrationReport(BaseModel):
    sample_count: int
    overall_rmse: float
    per_joint: dict[str, JointError]
    params: dict[str, float]

class CalibrationParams(BaseModel):
    scale_x: float
    scale_y: float
    scale_z: float
    sample_count: int

# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------
def mp_to_scene(lm_x: float, lm_y: float, lm_z: float) -> tuple[float, float, float]:
    """Replicate frontend mpToScene() using current _params."""
    x = (lm_x - 0.5) * _params["scale_x"]
    y = (1.0 - lm_y) * _params["scale_y"]
    z = lm_z * _params["scale_z"]
    return x, y, z

def update_last_landmarks(landmarks: list[dict], timestamp: float) -> None:
    """Called from pose.py after WebSocket send to store landmarks for comparison."""
    global _last_landmarks, _last_landmarks_ts
    _last_landmarks = landmarks
    _last_landmarks_ts = timestamp

# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------
def _optimize_params() -> None:
    """Least-squares regression to find optimal scale factors.

    Model:  obs_x = (lm.x - 0.5) * scale_x
    Solution: scale_x = sum((lm.x-0.5) * obs_x) / sum((lm.x-0.5)^2)
    (same for Y and Z axes)
    """
    global _params

    sum_xy = [0.0, 0.0, 0.0]  # numerator for x, y, z
    sum_xx = [0.0, 0.0, 0.0]  # denominator for x, y, z

    for s in _opt_samples:
        # X-axis: predictor = (lm.x - 0.5)
        px = s["lm_x"] - 0.5
        sum_xy[0] += px * s["obs_x"]
        sum_xx[0] += px * px

        # Y-axis: predictor = (1.0 - lm.y)
        py = 1.0 - s["lm_y"]
        sum_xy[1] += py * s["obs_y"]
        sum_xx[1] += py * py

        # Z-axis: predictor = lm.z
        pz = s["lm_z"]
        sum_xy[2] += pz * s["obs_z"]
        sum_xx[2] += pz * pz

    new_params = {}
    for i, axis in enumerate(["scale_x", "scale_y", "scale_z"]):
        if sum_xx[i] > 0.001:
            val = sum_xy[i] / sum_xx[i]
            val = max(0.5, min(6.0, val))  # clamp to reasonable range
        else:
            val = _params[axis]
        new_params[axis] = val

    # Enforce aspect ratio: no axis more than 2x any other axis.
    # This prevents the optimizer from drifting to extreme ratios (e.g. 1:4:1)
    # that break the model's built-in proportions.
    vals = [new_params["scale_x"], new_params["scale_y"], new_params["scale_z"]]
    v_max = max(vals)
    v_min = min(vals)
    if v_min > 0 and v_max / v_min > 2.0:
        # Pull the extreme axes toward the median
        target_ratio = 2.0
        for i, axis in enumerate(["scale_x", "scale_y", "scale_z"]):
            if vals[i] == v_max:
                new_params[axis] = v_min * target_ratio
            # Keep v_min and middle values as-is

    # Exponential moving average to prevent oscillation
    ema = 0.6
    for axis in ["scale_x", "scale_y", "scale_z"]:
        _params[axis] = round(ema * new_params[axis] + (1 - ema) * _params[axis], 3)

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@router.post("/frame")
async def accept_frame(report: FrameReport) -> dict:
    """Accept frontend 3D joint positions, compare against stored landmarks."""
    global _total_samples, _frames_since_opt, _opt_samples

    if _last_landmarks is None:
        return {"accepted": False, "reason": "no landmarks available"}

    # Timestamp must be within 0.5s of stored landmarks
    if abs(report.timestamp - _last_landmarks_ts) > 0.5:
        return {"accepted": False, "reason": "timestamp mismatch"}

    lm = _last_landmarks
    accepted_count = 0
    skipped_low_conf = 0
    skipped_unknown = 0

    for joint in report.joints:
        idx = JOINT_TO_IDX.get(joint.name)
        if idx is None:
            skipped_unknown += 1
            continue

        lm_data = lm[idx]
        if lm_data.get("visibility", 0) < 0.3:
            skipped_low_conf += 1
            continue

        accepted_count += 1

        # Target position via current mpToScene()
        tx, ty, tz = mp_to_scene(lm_data["x"], lm_data["y"], lm_data["z"])

        # Observed position from frontend
        ox, oy, oz = joint.x, joint.y, joint.z

        # Error
        dx = ox - tx
        dy = oy - ty
        dz = oz - tz
        err = (dx * dx + dy * dy + dz * dz) ** 0.5

        # Update stats
        if joint.name not in _stats:
            _stats[joint.name] = {"sum_sq": 0.0, "sum": 0.0, "n": 0}
        _stats[joint.name]["sum_sq"] += err * err
        _stats[joint.name]["sum"] += err
        _stats[joint.name]["n"] += 1

        # Store for optimization
        _opt_samples.append({
            "lm_x": lm_data["x"], "lm_y": lm_data["y"], "lm_z": lm_data["z"],
            "obs_x": ox, "obs_y": oy, "obs_z": oz,
            "joint_name": joint.name,
        })

    _total_samples += 1
    _frames_since_opt += 1

    # Log diagnostic on first few frames or periodically
    if _total_samples <= 5 or _total_samples % 30 == 0:
        print(f"[Calib] frame #{_total_samples}: {accepted_count} joints accepted, "
              f"skipped={skipped_low_conf} low-conf + {skipped_unknown} unknown "
              f"| total joints sent={len(report.joints)} | "
              f"ts_delta={abs(report.timestamp - _last_landmarks_ts):.3f}s", flush=True)

    # Trim opt samples to window
    max_samples = _opt_window * 13  # 13 joints per frame
    if len(_opt_samples) > max_samples:
        _opt_samples = _opt_samples[-max_samples:]

    # Run optimization every 50 accepted frames
    if _frames_since_opt >= 50 and len(_opt_samples) >= 130:
        _optimize_params()
        _frames_since_opt = 0

    return {"accepted": True, "samples": _total_samples}


@router.get("/report", response_model=CalibrationReport)
async def get_report() -> CalibrationReport:
    """Return current error statistics and calibration parameters."""
    per_joint = {}
    total_sum_sq = 0.0
    total_n = 0
    max_joint_n = 0

    for name, stat in _stats.items():
        n = stat["n"]
        if n == 0:
            continue
        if n > max_joint_n:
            max_joint_n = n
        rmse = (stat["sum_sq"] / n) ** 0.5
        mean_err = stat["sum"] / n
        per_joint[name] = JointError(rmse=round(rmse, 4), mean_error=round(mean_err, 4), sample_count=n)
        total_sum_sq += stat["sum_sq"]
        total_n += n

    overall_rmse = (total_sum_sq / total_n) ** 0.5 if total_n > 0 else 0.0

    return CalibrationReport(
        sample_count=max_joint_n,  # frames worth of data = max samples for any joint
        overall_rmse=round(overall_rmse, 4),
        per_joint=per_joint,
        params=dict(_params),
    )


@router.get("/params", response_model=CalibrationParams)
async def get_params() -> CalibrationParams:
    """Return current optimized scale factors."""
    return CalibrationParams(
        scale_x=_params["scale_x"],
        scale_y=_params["scale_y"],
        scale_z=_params["scale_z"],
        sample_count=_total_samples,
    )


@router.post("/reset")
async def reset() -> dict:
    """Clear all accumulated calibration statistics and reset params to defaults."""
    global _stats, _opt_samples, _total_samples, _frames_since_opt, _params
    _stats = {}
    _opt_samples = []
    _total_samples = 0
    _frames_since_opt = 0
    _params = {"scale_x": 2.5, "scale_y": 3.5, "scale_z": 2.5}
    return {"status": "reset"}
