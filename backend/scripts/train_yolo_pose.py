"""
YOLOv8-pose 训练脚本 —— 使用 MediaPipe 自动标注的数据进行知识蒸馏

用法:
    python -m scripts.train_yolo_pose

训练数据格式:
    data/train/images/   JPEG 图片
    data/train/labels/   YOLO 格式标签 (每个 .txt 一行: class cx cy w h kp1x kp1y kp1v ...)

输出:
    backend/models/yolo_pose_custom.pt
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT.parent / "data" / "train"
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "configs"
DEFAULT_OUTPUT = PROJECT_ROOT / "models" / "yolo_pose_custom.pt"


def _prepare_yolo_config(data_dir: Path, config_dir: Path) -> Path:
    """生成 YOLO 训练数据集配置文件"""
    config_path = config_dir / "yolo_data.yaml"

    config = f"""# YOLO pose training dataset (auto-generated)
path: {data_dir}
train: images
val: images

kpt_shape: [17, 3]
flip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]

names:
  0: person
"""
    config_path.write_text(config, encoding="utf-8")
    return config_path


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLOv8-pose 训练脚本")
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA_DIR), help="训练数据目录")
    parser.add_argument("--model", type=str, default="yolov8n-pose.pt", help="预训练模型")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="输出模型路径")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--imgsz", type=int, default=640, help="输入尺寸")
    parser.add_argument("--batch", type=int, default=16, help="批次大小")
    parser.add_argument("--lr", type=float, default=0.001, help="学习率")
    parser.add_argument("--device", type=str, default="auto", help="设备 (auto/cpu/cuda:0)")
    parser.add_argument("--resume", action="store_true", help="继续上次训练")
    args = parser.parse_args()

    data_dir = Path(args.data)
    config_dir = Path(args.config_dir)
    config_dir.mkdir(parents=True, exist_ok=True)

    # 统计数据集
    img_count = len(list((data_dir / "images").glob("*.jpg")))
    lbl_count = len(list((data_dir / "labels").glob("*.txt")))
    print(f"[Train] 数据集统计: {img_count} 张图片, {lbl_count} 个标签")

    if img_count < 10:
        print("[Train] 警告: 数据量过少 (< 10)，建议至少采集 200 帧")

    # 生成 YOLO 数据集配置
    data_config = _prepare_yolo_config(data_dir, config_dir)
    print(f"[Train] 数据集配置: {data_config}")

    # 加载预训练模型
    model = YOLO(args.model)

    # 训练
    results = model.train(
        data=str(data_config),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr,
        device=args.device,
        resume=args.resume,
        patience=20,
        save=True,
        save_period=10,
        exist_ok=True,
        pretrained=True,
        optimizer="auto",
        verbose=True,
        seed=42,
        # 数据增强
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
    )

    # 复制最佳模型到输出路径
    best_path = Path(results.save_dir) / "weights" / "best.pt"
    if best_path.exists():
        shutil.copy(str(best_path), str(args.output))
        print(f"[Train] 最佳模型已保存: {args.output}")
    else:
        print(f"[Train] 警告: 未找到最佳模型文件 {best_path}")

    # 导出为 ONNX (可选)
    try:
        loaded = YOLO(str(args.output))
        onnx_path = args.output.with_suffix(".onnx")
        loaded.export(format="onnx", imgsz=args.imgsz)
        print(f"[Train] ONNX 导出: {onnx_path}")
    except Exception as e:
        print(f"[Train] ONNX 导出失败 (非致命): {e}")


if __name__ == "__main__":
    main()
