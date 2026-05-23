"""
性能基准测试 —— 对比 MediaPipe / YOLO / Hybrid 三种后端

用法:
    python -m scripts.benchmark --video <path/to/video.mp4>
    python -m scripts.benchmark --camera 0  # 使用摄像头
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class BenchmarkRunner:
    """运行指定后端的基准测试"""

    def __init__(self, backend: str):
        self.backend = backend
        self.total_frames = 0
        self.detected_frames = 0
        self.total_time = 0.0
        self.latencies: list[float] = []
        self.last_positions: list[list[float]] | None = None
        self.jitter_sum = 0.0

    def _init_pipeline(self):
        if self.backend == "mediapipe":
            from app.services.pose_detection import CameraCapture, PoseDetector
            return _MediaPipeRunner(CameraCapture, PoseDetector)
        elif self.backend == "yolo":
            from app.services.yolo_detection import YoloPoseDetector
            return _YoloPoseRunner(YoloPoseDetector)
        elif self.backend == "hybrid":
            from app.services.yolo_detection import HybridPosePipeline
            return _HybridRunner(HybridPosePipeline)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def run_on_video(self, video_path: str) -> dict:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        self._run(cap)
        cap.release()
        return self._report()

    def run_on_camera(self, camera_index: int = 0, duration: float = 10.0) -> dict:
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        start = time.time()
        self._run(cap, max_duration=duration)
        cap.release()
        return self._report()

    def _run(self, cap, max_duration: float | None = None) -> None:
        from app.services.pose_detection import PoseDetector

        detector = PoseDetector()

        start_time = time.time()
        while True:
            if max_duration and (time.time() - start_time) > max_duration:
                break

            ret, frame = cap.read()
            if not ret:
                break

            self.total_frames += 1
            t0 = time.time()

            if self.backend == "mediapipe":
                pf = detector.process_frame(frame)
            elif self.backend == "yolo":
                from app.services.yolo_detection import YoloPoseDetector
                # Lazy init
                if not hasattr(self, '_yolo_det'):
                    self._yolo_det = YoloPoseDetector()
                pf = self._yolo_det.process_frame(frame)
            elif self.backend == "hybrid":
                # For video benchmark, detect person then crop + mediapipe
                from app.services.yolo_detection import YoloPersonDetector
                if not hasattr(self, '_person_det'):
                    self._person_det = YoloPersonDetector()
                bbox = self._person_det.detect(frame)
                if bbox:
                    pf = self._process_roi(frame, bbox, detector)
                else:
                    pf = detector.process_frame(frame)
            else:
                pf = None

            elapsed = time.time() - t0
            self.latencies.append(elapsed * 1000)

            if pf is not None:
                self.detected_frames += 1

                # 计算关键点抖动 (相邻帧位移)
                if self.last_positions is not None:
                    for i, lm in enumerate(pf.landmarks):
                        if i < len(self.last_positions):
                            dx = lm.x - self.last_positions[i][0]
                            dy = lm.y - self.last_positions[i][1]
                            self.jitter_sum += (dx * dx + dy * dy) ** 0.5

                self.last_positions = [[lm.x, lm.y] for lm in pf.landmarks]

    def _process_roi(self, frame, bbox, detector):
        """Hybrid mode: crop ROI, run MediaPipe, remap"""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        margin_x = int((x2 - x1) * 0.15)
        margin_y = int((y2 - y1) * 0.15)
        x1 = max(0, x1 - margin_x)
        y1 = max(0, y1 - margin_y)
        x2 = min(w, x2 + margin_x)
        y2 = min(h, y2 + margin_y)
        cropped = frame[y1:y2, x1:x2]

        pf = detector.process_frame(cropped)
        if pf is None:
            return None

        ch, cw = cropped.shape[:2]
        # Remap landmarks back to full frame
        for lm in pf.landmarks:
            px = lm.x * cw + x1
            py = lm.y * ch + y1
            lm.x = px / w
            lm.y = py / h

        pf.backend = "hybrid"
        return pf

    def _report(self) -> dict:
        if self.total_frames == 0:
            return {"backend": self.backend, "error": "No frames processed"}

        avg_latency = np.mean(self.latencies) if self.latencies else 0.0
        avg_fps = 1.0 / (avg_latency / 1000) if avg_latency > 0 else 0.0
        detect_rate = self.detected_frames / self.total_frames * 100
        avg_jitter = self.jitter_sum / max(self.detected_frames - 1, 1) if self.detected_frames > 1 else 0.0

        return {
            "backend": self.backend,
            "total_frames": self.total_frames,
            "detected_frames": self.detected_frames,
            "detection_rate": f"{detect_rate:.1f}%",
            "avg_latency_ms": f"{avg_latency:.2f}",
            "avg_fps": f"{avg_fps:.2f}",
            "avg_jitter": f"{avg_jitter:.6f}",
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="姿态检测后端性能对比")
    parser.add_argument("--video", type=str, default=None, help="测试视频路径")
    parser.add_argument("--camera", type=int, default=None, help="摄像头索引")
    parser.add_argument("--duration", type=float, default=10.0, help="摄像头测试时长(秒)")
    parser.add_argument("--backends", type=str, default="mediapipe,hybrid",
                        help="逗号分隔的后端列表: mediapipe,yolo,hybrid")
    args = parser.parse_args()

    if args.video is None and args.camera is None:
        print("[Benchmark] 请指定 --video 或 --camera")
        return

    backends = [b.strip() for b in args.backends.split(",")]
    results = []

    for backend in backends:
        print(f"\n[Benchmark] 测试 {backend} 后端...")
        runner = BenchmarkRunner(backend)

        if args.video:
            result = runner.run_on_video(args.video)
        else:
            result = runner.run_on_camera(args.camera, args.duration)

        results.append(result)
        print(f"  FPS={result.get('avg_fps','N/A')}  延迟={result.get('avg_latency_ms','N/A')}ms  检测率={result.get('detection_rate','N/A')}")

    # 汇总表格
    print("\n" + "=" * 80)
    print(f"{'Backend':<12} {'FPS':>8} {'Latency(ms)':>14} {'Detect Rate':>13} {'Jitter':>10}")
    print("-" * 80)
    for r in results:
        print(f"{r['backend']:<12} {r.get('avg_fps','N/A'):>8} {r.get('avg_latency_ms','N/A'):>14} "
              f"{r.get('detection_rate','N/A'):>13} {r.get('avg_jitter','N/A'):>10}")
    print("=" * 80)


if __name__ == "__main__":
    main()
