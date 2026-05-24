"""
Generate synthetic pose training data with stick figures.
Creates varied human poses on random backgrounds with perfect YOLO-format labels.

Usage:
    python -m scripts.generate_synthetic_data --count 500
"""
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT.parent / "data" / "train"

# COCO 17 keypoint skeleton connections
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # face
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # arms
    (5, 11), (6, 12), (11, 12),  # torso
    (11, 13), (13, 15), (12, 14), (14, 16),  # legs
]

# Default T-pose / neutral pose (17 COCO keypoints, normalized 0-1)
# Indices: nose(0), L-eye(1), R-eye(2), L-ear(3), R-ear(4),
#          L-sh(5), R-sh(6), L-elb(7), R-elb(8), L-wr(9), R-wr(10),
#          L-hip(11), R-hip(12), L-knee(13), R-knee(14), L-ank(15), R-ank(16)
NEUTRAL_POSE = np.array([
    [0.50, 0.10],  # 0  nose
    [0.47, 0.08],  # 1  L-eye
    [0.53, 0.08],  # 2  R-eye
    [0.44, 0.09],  # 3  L-ear
    [0.56, 0.09],  # 4  R-ear
    [0.38, 0.25],  # 5  L-sh
    [0.62, 0.25],  # 6  R-sh
    [0.28, 0.42],  # 7  L-elb
    [0.72, 0.42],  # 8  R-elb
    [0.22, 0.58],  # 9  L-wr
    [0.78, 0.58],  # 10 R-wr
    [0.42, 0.52],  # 11 L-hip
    [0.58, 0.52],  # 12 R-hip
    [0.40, 0.72],  # 13 L-knee
    [0.60, 0.72],  # 14 R-knee
    [0.38, 0.90],  # 15 L-ank
    [0.62, 0.90],  # 16 R-ank
], dtype=np.float32)

# Bone lengths (for constraining random poses)
BONE_PAIRS = [
    (5, 7, 0.17), (7, 9, 0.16),  # L arm
    (6, 8, 0.17), (8, 10, 0.16),  # R arm
    (5, 11, 0.27), (6, 12, 0.27),  # shoulder→hip
    (11, 13, 0.20), (13, 15, 0.18),  # L leg
    (12, 14, 0.20), (14, 16, 0.18),  # R leg
    (5, 6, 0.24),  # shoulder width
    (11, 12, 0.16),  # hip width
]

LATERAL_SYMMETRY = {
    1: 2, 2: 1, 3: 4, 4: 3,  # eyes, ears
    5: 6, 6: 5, 7: 8, 8: 7, 9: 10, 10: 9,  # arms
    11: 12, 12: 11, 13: 14, 14: 13, 15: 16, 16: 15,  # legs
}


def _rotate_point(px, py, cx, cy, angle_rad):
    """Rotate point (px,py) around center (cx,cy) by angle_rad."""
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    dx = px - cx
    dy = py - cy
    return cx + dx * cos_a - dy * sin_a, cy + dx * sin_a + dy * cos_a


def generate_pose(random_state: random.Random) -> np.ndarray:
    """Generate a randomized but plausible human pose (17 COCO keypoints)."""
    kp = NEUTRAL_POSE.copy()

    # Randomize joint angles using forward kinematics from root (hip center)
    hip_cx = (kp[11, 0] + kp[12, 0]) / 2
    hip_cy = (kp[11, 1] + kp[12, 1]) / 2
    sh_cx = (kp[5, 0] + kp[6, 0]) / 2
    sh_cy = (kp[5, 1] + kp[6, 1]) / 2

    # Torso lean
    lean_angle = random_state.uniform(-0.15, 0.15)
    for i in range(17):
        kp[i, 0], kp[i, 1] = _rotate_point(kp[i, 0], kp[i, 1], hip_cx, hip_cy, lean_angle)

    # Recompute after lean
    sh_cx = (kp[5, 0] + kp[6, 0]) / 2
    sh_cy = (kp[5, 1] + kp[6, 1]) / 2

    # Arms: rotate at shoulders
    l_arm_angle = random_state.uniform(-1.2, 0.8)  # up/down
    r_arm_angle = random_state.uniform(-1.2, 0.8)
    for idx in [7, 9]:  # L elbow, L wrist
        kp[idx, 0], kp[idx, 1] = _rotate_point(kp[idx, 0], kp[idx, 1], kp[5, 0], kp[5, 1], l_arm_angle)
    for idx in [8, 10]:  # R elbow, R wrist
        kp[idx, 0], kp[idx, 1] = _rotate_point(kp[idx, 0], kp[idx, 1], kp[6, 0], kp[6, 1], r_arm_angle)

    # Elbow bend
    l_elbow_angle = random_state.uniform(-0.8, 0.4)
    r_elbow_angle = random_state.uniform(-0.8, 0.4)
    kp[9, 0], kp[9, 1] = _rotate_point(kp[9, 0], kp[9, 1], kp[7, 0], kp[7, 1], l_elbow_angle)
    kp[10, 0], kp[10, 1] = _rotate_point(kp[10, 0], kp[10, 1], kp[8, 0], kp[8, 1], r_elbow_angle)

    # Legs: rotate at hips
    l_leg_angle = random_state.uniform(-0.6, 0.6)
    r_leg_angle = random_state.uniform(-0.6, 0.6)
    for idx in [13, 15]:
        kp[idx, 0], kp[idx, 1] = _rotate_point(kp[idx, 0], kp[idx, 1], kp[11, 0], kp[11, 1], l_leg_angle)
    for idx in [14, 16]:
        kp[idx, 0], kp[idx, 1] = _rotate_point(kp[idx, 0], kp[idx, 1], kp[12, 0], kp[12, 1], r_leg_angle)

    # Knee bend
    l_knee_angle = random_state.uniform(-0.3, 0.8)
    r_knee_angle = random_state.uniform(-0.3, 0.8)
    kp[15, 0], kp[15, 1] = _rotate_point(kp[15, 0], kp[15, 1], kp[13, 0], kp[13, 1], l_knee_angle)
    kp[16, 0], kp[16, 1] = _rotate_point(kp[16, 0], kp[16, 1], kp[14, 0], kp[14, 1], r_knee_angle)

    # Ensure keypoints stay in reasonable range
    kp = np.clip(kp, 0.02, 0.98)

    return kp


def _compute_bbox(kpts: np.ndarray) -> tuple[float, float, float, float]:
    """Compute YOLO-format bbox (cx, cy, w, h) from keypoints."""
    valid = kpts[kpts[:, 2] > 0.3] if kpts.shape[1] > 2 else kpts
    if len(valid) < 3:
        valid = kpts
    min_x, min_y = valid[:, 0].min(), valid[:, 1].min()
    max_x, max_y = valid[:, 0].max(), valid[:, 1].max()
    w = max(max_x - min_x, 0.05)
    h = max(max_y - min_y, 0.05)
    cx = (min_x + max_x) / 2
    cy = (min_y + max_y) / 2
    # Expand for half-body estimation
    w = min(w * 1.5, 0.98)
    h = min(h * 1.5, 0.98)
    return cx, cy, w, h


def draw_person(img: np.ndarray, kpts: np.ndarray, color: tuple) -> np.ndarray:
    """Draw a person stick figure on the image."""
    h, w = img.shape[:2]
    # Draw skeleton lines
    for a, b in SKELETON:
        if kpts[a, 2] > 0.3 and kpts[b, 2] > 0.3:
            pt1 = (int(kpts[a, 0] * w), int(kpts[a, 1] * h))
            pt2 = (int(kpts[b, 0] * w), int(kpts[b, 1] * h))
            cv2.line(img, pt1, pt2, color, 2, cv2.LINE_AA)
    # Draw joints
    for i in range(17):
        if kpts[i, 2] > 0.3:
            pt = (int(kpts[i, 0] * w), int(kpts[i, 1] * h))
            radius = 4 if i in [0, 5, 6, 11, 12] else 3
            cv2.circle(img, pt, radius, (0, 0, 255), -1, cv2.LINE_AA)
    return img


def generate_background(h: int, w: int, rng: random.Random) -> np.ndarray:
    """Generate a varied background."""
    bg_type = rng.randint(0, 3)
    if bg_type == 0:
        # Solid color with noise
        color = (rng.randint(30, 180), rng.randint(30, 180), rng.randint(30, 180))
        bg = np.full((h, w, 3), color, dtype=np.uint8)
        noise = np.random.randint(0, 30, (h, w, 3), dtype=np.uint8)
        bg = cv2.add(bg, noise)
    elif bg_type == 1:
        # Gradient
        c1 = np.array([rng.randint(20, 150), rng.randint(20, 150), rng.randint(20, 150)])
        c2 = np.array([rng.randint(30, 180), rng.randint(30, 180), rng.randint(30, 180)])
        bg = np.zeros((h, w, 3), dtype=np.uint8)
        for y in range(h):
            t = y / h
            color = (c1 * (1 - t) + c2 * t).astype(np.uint8)
            bg[y, :] = color
        noise = np.random.randint(0, 25, (h, w, 3), dtype=np.uint8)
        bg = cv2.add(bg, noise)
    else:
        # Random pattern
        bg = np.random.randint(20, 180, (h, w, 3), dtype=np.uint8)
        bg = cv2.GaussianBlur(bg, (21, 21), 10)

    return bg


def generate_sample(rng: random.Random, img_w: int = 640, img_h: int = 640) -> tuple[np.ndarray, np.ndarray]:
    """Generate a single training sample (image + 17 COCO keypoints with visibility)."""
    bg = generate_background(img_h, img_w, rng)

    # Generate main pose
    kpts_xy = generate_pose(rng)
    kpts = np.ones((17, 3), dtype=np.float32)
    kpts[:, :2] = kpts_xy
    kpts[:, 2] = 1.0  # all visible

    # Randomize body position in frame
    offset_x = rng.uniform(-0.08, 0.08)
    offset_y = rng.uniform(-0.05, 0.05)
    kpts[:, 0] += offset_x
    kpts[:, 1] += offset_y

    # Scale body
    scale = rng.uniform(0.7, 1.2)
    cx = kpts[:, 0].mean()
    cy = kpts[:, 1].mean()
    kpts[:, 0] = cx + (kpts[:, 0] - cx) * scale
    kpts[:, 1] = cy + (kpts[:, 1] - cy) * scale

    # Clip to valid range
    kpts[:, 0] = np.clip(kpts[:, 0], 0.02, 0.98)
    kpts[:, 1] = np.clip(kpts[:, 1], 0.02, 0.98)

    # Draw main person
    person_color = (rng.randint(50, 200), rng.randint(100, 220), rng.randint(50, 200))
    img = draw_person(bg, kpts, person_color)

    # Sometimes add a second (smaller, background) person
    if rng.random() < 0.2:
        kpts2_xy = generate_pose(rng)
        kpts2 = np.ones((17, 3), dtype=np.float32)
        kpts2[:, :2] = kpts2_xy * 0.6
        kpts2[:, 0] += rng.uniform(0.2, 0.5)
        kpts2[:, 1] += rng.uniform(0.1, 0.3)
        kpts2[:, 2] = np.random.choice([0.0, 1.0], size=17, p=[0.3, 0.7])
        draw_person(img, kpts2, (100, 100, 100))

    # Add blur and JPEG-like artifacts
    if rng.random() < 0.3:
        ksize = rng.choice([3, 5])
        img = cv2.GaussianBlur(img, (ksize, ksize), 0)

    return img, kpts


def save_sample(img: np.ndarray, kpts: np.ndarray, img_dir: Path, lbl_dir: Path, idx: int) -> None:
    """Save image and YOLO-format label."""
    img_path = img_dir / f"synthetic_{idx:06d}.jpg"
    lbl_path = lbl_dir / f"synthetic_{idx:06d}.txt"

    cv2.imwrite(str(img_path), img)

    # YOLO format: class cx cy w h kp1x kp1y kp1v ...
    cx, cy, w, h = _compute_bbox(kpts)
    parts = [f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"]
    for kx, ky, kv in kpts:
        parts.append(f"{kx:.6f} {ky:.6f} {kv:.6f}")

    lbl_path.write_text(" ".join(parts))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic pose training data")
    parser.add_argument("--count", type=int, default=500, help="Number of samples to generate")
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR), help="Output directory")
    parser.add_argument("--img-size", type=int, default=640, help="Image size (square)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.output)
    img_dir = out_dir / "images"
    lbl_dir = out_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(exist_ok=True)

    print(f"[SyntheticData] 生成 {args.count} 张合成训练数据...")
    for i in range(args.count):
        img, kpts = generate_sample(rng, args.img_size, args.img_size)
        save_sample(img, kpts, img_dir, lbl_dir, i)
        if (i + 1) % 100 == 0:
            print(f"[SyntheticData]   {i + 1}/{args.count} 已生成")

    print(f"[SyntheticData] 完成! 共 {args.count} 张图片 + 标签")
    print(f"[SyntheticData] 图片目录: {img_dir}")
    print(f"[SyntheticData] 标签目录: {lbl_dir}")


if __name__ == "__main__":
    main()
