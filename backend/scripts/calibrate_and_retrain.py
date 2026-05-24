#!/usr/bin/env python3
"""
Continuous calibration + retraining loop.

Usage:
    python -m scripts.calibrate_and_retrain --min-samples 100 --cycles 3

Flow per cycle:
  1. Poll GET /calibrate/report until N samples collected or timeout
  2. Fetch GET /calibrate/params for optimal scale factors
  3. Save params to calibration_params.json
  4. Regenerate synthetic training data
  5. Retrain YOLOv8n-pose model
  6. Benchmark new model
  7. If RMSE improvement < threshold, stop; else POST /calibrate/reset and loop
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent.parent
PARAMS_PATH = ROOT / "app" / "core" / "calibration_params.json"
MODEL_PATH = ROOT / "models" / "yolo_pose_custom.pt"
DATA_DIR = ROOT.parent / "data"
CONFIG_PATH = ROOT / "configs" / "custom_pose.yaml"

BASE_URL = "http://localhost:8000"


def poll_calibration(base_url: str, min_samples: int, poll_interval: float, timeout: float) -> dict | None:
    """Poll GET /calibrate/report until sample_count >= min_samples or timeout."""
    deadline = time.time() + timeout
    last_count = 0
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/calibrate/report", timeout=5) as resp:
                data = json.loads(resp.read())
            count = data.get("sample_count", 0)
            rmse = data.get("overall_rmse", 0)
            if count >= min_samples:
                print(f"[CalLoop] 采集完成: {count} 样本, RMSE={rmse:.4f}")
                return data
            if count != last_count:
                print(f"[CalLoop] 采集中... {count}/{min_samples} 样本 (RMSE={rmse:.4f})")
                last_count = count
        except Exception as e:
            print(f"[CalLoop] 轮询错误: {e}")
        time.sleep(poll_interval)
    print(f"[CalLoop] 超时: 仅采集到 {last_count}/{min_samples} 样本")
    return None


def fetch_params(base_url: str) -> dict | None:
    """GET /calibrate/params and return {scale_x, scale_y, scale_z}."""
    try:
        with urllib.request.urlopen(f"{base_url}/calibrate/params", timeout=5) as resp:
            return json.loads(resp.read())
    except Exception as e:
        print(f"[CalLoop] 获取参数失败: {e}")
        return None


def save_params(params: dict, path: Path) -> None:
    """Persist calibration params to JSON."""
    data = {
        "scale_x": params.get("scale_x", 2.5),
        "scale_y": params.get("scale_y", 3.5),
        "scale_z": params.get("scale_z", 2.5),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "sample_count": params.get("sample_count", 0),
    }
    path.write_text(json.dumps(data, indent=2))
    print(f"[CalLoop] 参数已保存: {path}")


def reset_calibration(base_url: str) -> None:
    """POST /calibrate/reset."""
    try:
        req = urllib.request.Request(f"{base_url}/calibrate/reset", method="POST")
        urllib.request.urlopen(req, timeout=5)
        print("[CalLoop] 校准统计已重置")
    except Exception as e:
        print(f"[CalLoop] 重置失败: {e}")


def generate_data(count: int = 500) -> bool:
    """Run generate_synthetic_data.py."""
    script = ROOT / "scripts" / "generate_synthetic_data.py"
    out = ROOT.parent / "data" / "train"
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(script),
        "--count", str(count),
        "--output", str(out),
        "--img-size", "640",
    ]
    print(f"[CalLoop] 生成合成数据: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    if result.returncode != 0:
        print(f"[CalLoop] 数据生成失败: {result.stderr[:500]}")
        return False
    print(f"[CalLoop] 数据生成完成")
    return True


def train_model() -> bool:
    """Retrain YOLOv8n-pose with existing config."""
    from ultralytics import YOLO

    # Start from the existing best model or from scratch
    if MODEL_PATH.exists():
        print(f"[CalLoop] 加载现有模型: {MODEL_PATH}")
        model = YOLO(str(MODEL_PATH))
    else:
        print("[CalLoop] 加载基础模型: yolov8n-pose.pt")
        model = YOLO("yolov8n-pose.pt")

    data_yaml = str(CONFIG_PATH)
    print(f"[CalLoop] 开始训练... data={data_yaml}")
    try:
        results = model.train(
            data=data_yaml,
            epochs=20,
            imgsz=640,
            batch=8,
            device=0,
            workers=2,
            name="custom_pose_retrain",
            exist_ok=True,
            verbose=False,
        )
        # Save best model — use actual save_dir from training results
        save_dir = getattr(results, "save_dir", None) or (ROOT / "runs" / "pose" / "custom_pose_retrain")
        best_src = Path(save_dir) / "weights" / "best.pt"
        if best_src.exists():
            import shutil
            shutil.copy(str(best_src), str(MODEL_PATH))
            print(f"[CalLoop] 模型已保存: {MODEL_PATH}")
            return True
        return False
    except Exception as e:
        print(f"[CalLoop] 训练失败: {e}")
        return False


def benchmark_model() -> dict:
    """Simple benchmark: check validation metrics."""
    # Check if the training run produced val results
    results_dir = ROOT / "runs" / "pose" / "custom_pose_retrain"
    results_csv = results_dir / "results.csv"
    metrics = {
        "pose_mAP50": 0.0,
        "pose_mAP50_95": 0.0,
    }
    if results_csv.exists():
        lines = results_csv.read_text().strip().split("\n")
        if len(lines) > 1:
            # Last line has latest epoch metrics
            last = lines[-1].strip().split(",")
            # Column layout depends on ultralytics version; try to find metric columns
            header = lines[0].strip().split(",")
            try:
                for i, col in enumerate(header):
                    col_clean = col.strip()
                    if "metrics/mAP50(B)" in col_clean:
                        metrics["pose_mAP50"] = float(last[i])
                    if "metrics/mAP50-95(B)" in col_clean:
                        metrics["pose_mAP50_95"] = float(last[i])
            except (ValueError, IndexError):
                pass
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Continuous calibration + retraining loop")
    parser.add_argument("--min-samples", type=int, default=100, help="Samples to collect before training")
    parser.add_argument("--cycles", type=int, default=3, help="Maximum calibration cycles")
    parser.add_argument("--timeout", type=float, default=600, help="Seconds to wait per cycle for samples")
    parser.add_argument("--poll-interval", type=float, default=3, help="Seconds between poll checks")
    parser.add_argument("--data-count", type=int, default=500, help="Synthetic samples per cycle")
    parser.add_argument("--base-url", type=str, default=BASE_URL, help="Backend base URL")
    args = parser.parse_args()

    prev_rmse = float("inf")

    for cycle in range(1, args.cycles + 1):
        print(f"\n{'='*60}")
        print(f"  校准周期 {cycle}/{args.cycles}")
        print(f"{'='*60}")

        # 1. Collect calibration data
        print(f"\n[Step 1/{cycle}] 采集校准数据 (最少 {args.min_samples} 样本)...")
        report = poll_calibration(args.base_url, args.min_samples, args.poll_interval, args.timeout)
        if report is None:
            print("[CalLoop] 未采集到足够数据, 退出")
            break

        current_rmse = report.get("overall_rmse", 0)
        print(f"[CalLoop] 当前 RMSE: {current_rmse:.4f} (上一轮: {prev_rmse:.4f})")

        # 2. Fetch optimized params
        print(f"\n[Step 2/{cycle}] 获取优化参数...")
        params = fetch_params(args.base_url)
        if params is None:
            break
        print(f"[CalLoop] scale_x={params.get('scale_x')}, scale_y={params.get('scale_y')}, scale_z={params.get('scale_z')}")

        # 3. Save params for server restart persistence
        print(f"\n[Step 3/{cycle}] 保存校准参数...")
        save_params(params, PARAMS_PATH)

        # 4. Regenerate synthetic data
        print(f"\n[Step 4/{cycle}] 重新生成合成训练数据...")
        if not generate_data(args.data_count):
            print("[CalLoop] 数据生成失败, 跳过本周期训练")
            break

        # 5. Retrain model
        print(f"\n[Step 5/{cycle}] 重新训练 YOLOv8n-pose...")
        if not train_model():
            print("[CalLoop] 训练失败")
            break

        # 6. Benchmark
        print(f"\n[Step 6/{cycle}] 评估模型...")
        metrics = benchmark_model()
        print(f"[CalLoop] 模型指标: {metrics}")

        # 7. Check convergence
        improvement = prev_rmse - current_rmse
        print(f"[CalLoop] RMSE 改进: {improvement:.4f} (阈值: {args.min_samples * 0.001:.4f})")
        if improvement < 0.001 and cycle > 1:
            print("[CalLoop] 收敛, 停止循环")
            break

        prev_rmse = current_rmse

        # Reset for next cycle
        print(f"\n[Step 7/{cycle}] 重置校准统计...")
        reset_calibration(args.base_url)

    print(f"\n{'='*60}")
    print(f"  校准完成! 模型: {MODEL_PATH}")
    print(f"  参数: {PARAMS_PATH}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
