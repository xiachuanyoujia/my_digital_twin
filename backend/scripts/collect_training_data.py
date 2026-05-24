"""
数据采集工具 —— 使用 MediaPipe 自动标注训练数据

用法:
    python -m scripts.collect_training_data

操作:
    s    保存当前帧 (含 MediaPipe 标注)
    q    退出

输出:
    data/train/images/   JPEG 图片
    data/train/labels/   YOLO 格式标签 (bbox + 17 COCO 关键点)
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options as bo
from mediapipe.tasks.python.vision import RunningMode

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "pose_landmarker_lite.task"
OUTPUT_DIR = PROJECT_ROOT.parent / "data" / "train"

HALF_BODY_RATIO = 3.0
COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


def _init_mediapipe() -> vision.PoseLandmarker:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"MediaPipe model not found: {MODEL_PATH}")
    base = bo.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = vision.PoseLandmarkerOptions(
        base_options=base,
        running_mode=RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.PoseLandmarker.create_from_options(options)


def _landmarks_to_coco(full_landmarks, frame_w: int, frame_h: int) -> list[tuple[float, float, float]]:
    """将 MediaPipe 33 个关键点映射为 17 个 COCO 关键点，返回归一化坐标 (x, y, visibility)"""
    # MediaPipe → COCO 映射:
    #   COCO 0 (nose) = MP 0
    #   COCO 1 (left_eye) = MP 2
    #   COCO 2 (right_eye) = MP 5
    #   COCO 3 (left_ear) = MP 7
    #   COCO 4 (right_ear) = MP 8
    #   COCO 5 (left_shoulder) = MP 11
    #   COCO 6 (right_shoulder) = MP 12
    #   COCO 7 (left_elbow) = MP 13
    #   COCO 8 (right_elbow) = MP 14
    #   COCO 9 (left_wrist) = MP 15
    #   COCO 10 (right_wrist) = MP 16
    #   COCO 11 (left_hip) = MP 23
    #   COCO 12 (right_hip) = MP 24
    #   COCO 13 (left_knee) = MP 25
    #   COCO 14 (right_knee) = MP 26
    #   COCO 15 (left_ankle) = MP 27
    #   COCO 16 (right_ankle) = MP 28
    mp_to_coco = {
        0: 0, 2: 1, 5: 2, 7: 3, 8: 4,
        11: 5, 12: 6, 13: 7, 14: 8,
        15: 9, 16: 10, 23: 11, 24: 12,
        25: 13, 26: 14, 27: 15, 28: 16,
    }

    result = []
    for coco_idx in range(17):
        mp_idx = None
        for mp_i, ci in mp_to_coco.items():
            if ci == coco_idx:
                mp_idx = mp_i
                break
        if mp_idx is not None and mp_idx < len(full_landmarks):
            lm = full_landmarks[mp_idx]
            # MediaPipe can return coords slightly outside [0,1] for body parts
            # partially off-frame (e.g. feet below image, hands above head)
            x = max(0.0, min(1.0, lm.x))
            y = max(0.0, min(1.0, lm.y))
            result.append((x, y, lm.visibility or 1.0))
        else:
            result.append((0.0, 0.0, 0.0))
    return result


def _compute_bbox(kpts: list[tuple[float, float, float]],
                  frame_w: int, frame_h: int) -> tuple[float, float, float, float]:
    """从关键点计算 YOLO 格式的边界框 (cx, cy, w, h) 归一化"""
    valid = [(k[0], k[1]) for k in kpts if k[2] > 0.3]
    if not valid:
        return (0.5, 0.5, 1.0, 1.0)

    xs = [p[0] for p in valid]
    ys = [p[1] for p in valid]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    cx = (min_x + max_x) / 2.0
    cy = (min_y + max_y) / 2.0
    w = max(max_x - min_x, 0.05)
    h = max(max_y - min_y, 0.05)

    # 半身扩展
    w = min(w * HALF_BODY_RATIO, 1.0)
    h = min(h * HALF_BODY_RATIO, 1.0)
    cx = max(w / 2, min(1.0 - w / 2, cx))
    cy = max(h / 2, min(1.0 - h / 2, cy))

    return (cx, cy, w, h)


def _draw_overlay(frame: np.ndarray, kpts: list[tuple[float, float, float]],
                  bbox: tuple[float, float, float, float],
                  saved_count: int, fps: float) -> np.ndarray:
    """绘制关键点、骨架和边界框"""
    h, w = frame.shape[:2]
    cx, cy, bw, bh = bbox
    x1 = int((cx - bw / 2) * w)
    y1 = int((cy - bh / 2) * h)
    x2 = int((cx + bw / 2) * w)
    y2 = int((cy + bh / 2) * h)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    for kp in kpts:
        if kp[2] > 0.3:
            px, py = int(kp[0] * w), int(kp[1] * h)
            cv2.circle(frame, (px, py), 3, (0, 0, 255), -1)

    # 骨架连线
    skeleton = [
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
        (12, 14), (14, 16),
    ]
    for a, b in skeleton:
        if kpts[a][2] > 0.3 and kpts[b][2] > 0.3:
            p1 = (int(kpts[a][0] * w), int(kpts[a][1] * h))
            p2 = (int(kpts[b][0] * w), int(kpts[b][1] * h))
            cv2.line(frame, p1, p2, (255, 255, 0), 1)

    cv2.putText(frame, f"Saved: {saved_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, "S: Save  Q: Quit", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    return frame


def _keypoints_changed(prev_kpts: list[tuple[float, float, float]],
                      curr_kpts: list[tuple[float, float, float]],
                      threshold: float = 0.03) -> bool:
    """Check if keypoints changed enough to warrant saving a new frame."""
    if prev_kpts is None:
        return True
    total_movement = 0.0
    valid = 0
    for pk, ck in zip(prev_kpts, curr_kpts):
        if pk[2] > 0.3 and ck[2] > 0.3:
            total_movement += ((ck[0] - pk[0]) ** 2 + (ck[1] - pk[1]) ** 2) ** 0.5
            valid += 1
    if valid == 0:
        return True
    return (total_movement / valid) > threshold


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO 姿态训练数据采集工具")
    parser.add_argument("--camera", type=int, default=0, help="摄像头索引")
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--auto", action="store_true",
                        help="自动采集模式: 人体检测到且姿态变化时自动保存")
    parser.add_argument("--interval", type=float, default=0.5,
                        help="自动采集最小间隔(秒), 默认0.5")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="自动采集最大帧数, 0=无限")
    args = parser.parse_args()

    out_dir = Path(args.output)
    img_dir = out_dir / "images"
    lbl_dir = out_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(exist_ok=True)

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    landmarker = _init_mediapipe()
    saved_count = len(list(img_dir.glob("*.jpg")))
    print(f"[DataCollector] 摄像头已启动，已保存 {saved_count} 帧")
    if args.auto:
        print(f"[DataCollector] 自动采集模式: 间隔={args.interval}s, 最大={args.max_frames or '无限'}")
    else:
        print("[DataCollector] S = 保存当前帧 | Q = 退出")

    prev_time = time.time()
    fps = 0.0
    last_auto_save = 0.0
    prev_kpts = None
    auto_saved = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            curr_time = time.time()
            dt = curr_time - prev_time
            prev_time = curr_time
            fps = fps * 0.9 + (1.0 / max(dt, 0.001)) * 0.1

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_image)

            display = frame.copy()
            bbox = (0.5, 0.5, 1.0, 1.0)
            kpts = [(0.0, 0.0, 0.0)] * 17

            if result.pose_landmarks:
                full = result.pose_landmarks[0]
                kpts = _landmarks_to_coco(full, args.width, args.height)
                bbox = _compute_bbox(kpts, args.width, args.height)

            should_save = False

            if args.auto and result.pose_landmarks:
                if args.max_frames > 0 and auto_saved >= args.max_frames:
                    print(f"[DataCollector] 已达到最大采集帧数 {args.max_frames}, 退出")
                    break
                if (curr_time - last_auto_save) >= args.interval:
                    if _keypoints_changed(prev_kpts, kpts):
                        should_save = True

            display = _draw_overlay(display, kpts, bbox, saved_count, fps)
            if args.auto:
                cv2.putText(display, f"Auto: {auto_saved}/{args.max_frames or 'inf'}",
                            (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 1)
            cv2.imshow("Data Collector", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and not args.auto:
                should_save = True

            if should_save:
                ts = int(time.time() * 1000)
                img_path = img_dir / f"frame_{ts}.jpg"
                lbl_path = lbl_dir / f"frame_{ts}.txt"

                cv2.imwrite(str(img_path), frame)

                # YOLO 格式: class cx cy w h kp1x kp1y kp1v ...
                cx, cy, bw_, bh_ = bbox
                label_parts = [f"0 {cx:.6f} {cy:.6f} {bw_:.6f} {bh_:.6f}"]
                for kx, ky, kv in kpts:
                    label_parts.append(f"{kx:.6f} {ky:.6f} {kv:.6f}")

                lbl_path.write_text(" ".join(label_parts))

                saved_count += 1
                auto_saved += 1
                prev_kpts = kpts
                last_auto_save = curr_time
                print(f"[DataCollector] 已保存 #{saved_count}: {img_path.name}")

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        landmarker.close()
        cv2.destroyAllWindows()
        print(f"[DataCollector] 共计保存 {saved_count} 帧")


if __name__ == "__main__":
    main()
