# YOLOv8 集成方案

## 背景

当前项目使用 MediaPipe PoseLandmarker Lite 进行人体姿态检测，实时驱动 3D 机甲模型。存在的问题：MediaPipe 在高分辨率下处理全帧（1280×720）会导致帧率下降，且没有针对人物区域进行裁剪优化，背景噪声影响检测精度。

**目标**：引入 YOLOv8 作为**人物检测前置阶段**——先检测人物边界框并裁剪，再对裁剪后的区域运行 MediaPipe。这样既能提升速度（输入尺寸更小），也能提升精度（减少背景干扰）。同时增加 **YOLOv8-pose 端到端模式**作为更快速的替代方案，通过知识蒸馏从 MediaPipe 学习。

## 推荐方案

**两级流水线（主要模式）**：YOLOv8n 人物检测 → 裁剪 ROI → MediaPipe PoseLandmarker → 13 个关键点 → WebSocket  
**单级流水线（替代模式）**：YOLOv8n-pose → 17 个关键点 → 取 13 个关键点子集 → WebSocket

项目使用的 13 个关键点（鼻子、双肩、双肘、双腕、双髋、双膝、双踝）恰好是 COCO 17 个关键点的子集（仅去掉了眼睛和耳朵），因此 YOLOv8-pose 可以开箱即用，前端无需任何修改。

---

## 实施步骤

### 第一步：添加 YOLO 依赖

**文件**：`backend/requirements.txt`  
新增依赖：`ultralytics>=8.3.0`、`torch>=2.5.0`

执行 `pip install ultralytics torch` 安装。

### 第二步：创建 YOLO 检测服务

**新文件**：`backend/app/services/yolo_detection.py`

包含以下类：

- **`YoloPersonDetector`**：封装 YOLOv8n（COCO 预训练），检测人物边界框，返回最大人物的 bbox（x1,y1,x2,y2）和置信度
- **`YoloPoseDetector`**：封装 YOLOv8n-pose，返回 17 个 COCO 关键点，映射为项目需要的 13 个 Landmark 格式
- **`HybridPosePipeline`**：组合 YOLO 人物检测 + MediaPipe 姿态估计（在裁剪后的 ROI 上运行），如果 YOLO 未检测到人物则降级为全帧 MediaPipe

关键设计：
- 延迟加载模型（首次推理时才加载 .pt 文件）
- GPU 自动检测（RTX 5060 Ti 上使用 CUDA 加速）
- 输出格式与现有 `PoseFrame` 完全一致——前端无感知
- 复用现有的 `PoseSmoother` 进行时序滤波

### 第三步：更新配置

**文件**：`backend/app/core/config.py`

新增字段：
```python
pose_backend: str = "hybrid"      # "mediapipe" | "yolo" | "hybrid"
yolo_person_model: str = "yolov8n.pt"       # 人物检测模型
yolo_pose_model: str = "yolov8n-pose.pt"    # 姿态估计模型
yolo_confidence: float = 0.5
yolo_iou: float = 0.45
```

### 第四步：创建数据采集与训练脚本

**新文件**：`backend/scripts/collect_training_data.py`
- 打开摄像头，运行 MediaPipe PoseLandmarker
- 将帧保存为 JPEG + YOLO 格式标签（从关键点推算出边界框 + 17 个关键点坐标）
- 输出目录：`data/train/images/` + `data/train/labels/`
- 交互方式：按 `s` 保存当前帧，按 `q` 退出
- 实时预览画面并叠加关键点标注

**新文件**：`backend/scripts/train_yolo_pose.py`
- 加载采集的数据集
- 训练 YOLOv8n-pose（可选择 YOLOv8s-pose 以获得更高精度）
- 使用 MediaPipe 自动标注的数据进行知识蒸馏
- 训练好的模型保存至 `backend/models/yolo_pose_custom.pt`
- 包含验证集划分、早停机制、数据增强配置

**新文件**：`backend/configs/yolo_train.yaml`
- 训练超参数：epochs=100, imgsz=640, batch=16, lr0=0.01
- 数据增强：mosaic、flip、HSV 色彩抖动
- 早停 patience=20

### 第五步：更新 API 路由以支持模型切换

**文件**：`backend/app/api/routes/pose.py`
- `get_pipeline()` 根据 `settings.pose_backend` 实例化对应的流水线类
- 新增 `POST /pose/config` 接口支持运行时切换后端（便于 A/B 对比测试）
- 更新 `/status` 接口，返回当前使用的后端类型及实际 FPS/延迟

### 第六步：更新前端状态展示

**文件**：`frontend/src/main.js`
- 在状态指示器中展示当前使用的检测后端
- 通过 WebSocket 接收后端类型信息

### 第七步：添加性能对比脚本

**新文件**：`backend/scripts/benchmark.py`
- 对同一段录制视频分别运行三种后端（mediapipe、yolo、hybrid）
- 报告指标：平均 FPS、平均延迟、检测成功率、关键点稳定性（抖动程度）
- 在控制台输出对比表格

---

## 文件变更汇总

| 操作 | 文件 | 说明 |
|------|------|------|
| 新增 | `backend/app/services/yolo_detection.py` | YOLO 人物/姿态检测服务 |
| 新增 | `backend/scripts/collect_training_data.py` | 自动标注数据采集工具 |
| 新增 | `backend/scripts/train_yolo_pose.py` | YOLOv8-pose 训练脚本 |
| 新增 | `backend/configs/yolo_train.yaml` | 训练配置文件 |
| 新增 | `backend/scripts/benchmark.py` | 速度/精度对比工具 |
| 修改 | `backend/requirements.txt` | 新增 ultralytics、torch |
| 修改 | `backend/app/core/config.py` | 新增 YOLO 配置项 |
| 修改 | `backend/app/api/routes/pose.py` | 模型切换 + 完善状态上报 |
| 修改 | `backend/app/models/pose.py` | 新增后端类型枚举 |
| 修改 | `frontend/src/main.js` | UI 展示当前后端类型 |

## 数据流（混合模式）

```
摄像头 → YOLOv8n (人物检测 bbox) → 裁剪 ROI → MediaPipe PoseLandmarker → 13 个关键点 → PoseSmoother → PoseFrame → WebSocket JSON → Three.js 机甲模型
         ↓ (未检测到人物时降级)
         全帧 MediaPipe
```

## 验证方案

1. **性能测试**：对录制的视频运行 `benchmark.py`，验证混合模式 FPS 高于纯 MediaPipe 模式
2. **集成测试**：后端以 `pose_backend=hybrid` 启动，打开前端，验证机甲模型正确跟随人体动作
3. **训练验证**：使用 `collect_training_data.py` 采集 200+ 帧数据，训练 YOLOv8n-pose，确认损失收敛
4. **A/B 对比**：通过 API 在不同后端间切换，确认均能正确驱动机甲
5. **边界情况**：测试画面中无人物（降级逻辑）、部分遮挡、暗光环境
