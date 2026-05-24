# My Digital Twin 项目成果报告

> 版本: 0.1.0 | 日期: 2026-05-24 | 分支: dev

---

## 一、项目概述

**My Digital Twin** 是一款实时人体姿态驱动的 3D 数字孪生应用。通过摄像头实时捕捉人体动作，利用深度学习模型进行姿态估计，将骨骼关键点数据通过 WebSocket 实时传输至前端，驱动 3D 机甲模型实现真人与虚拟角色的实时同步。

### 核心指标达成

| 指标 | 目标值 | 当前状态 |
|------|--------|----------|
| 姿态检测帧率 | ≥ 30fps | 达标 (三种后端均满足) |
| 渲染帧率 | ≥ 60fps | 达标 (Three.js WebGL) |
| 端到端延迟 | < 50ms | 达标 |
| 关键点数量 | 13 个 | 达标 (覆盖全身主要关节) |
| 检测后端 | 1 种 | 超额 (3 种可切换) |

---

## 二、技术架构

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                        摄像头 (USB/内置)                          │
│                    1280×720 @ 60fps 采集                         │
└─────────────────────────┬───────────────────────────────────────┘
                          │ BGR Frame
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI 后端 (Python 3.11+)                    │
│                                                                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                  姿态检测引擎 (三选一)                      │  │
│  │                                                             │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │  │
│  │  │  MediaPipe   │  │ YOLOv8-pose  │  │   混合模式    │    │  │
│  │  │   (纯MP)     │  │   (纯YOLO)   │  │ YOLO+MediaPipe│    │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘    │  │
│  │                                                             │  │
│  │  输出: 13个3D关键点 + 世界坐标 + 可见度                      │  │
│  └───────────────────────────────────────────────────────────┘  │
│                          │                                       │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              PoseSmoother (自适应时序滤波)                   │  │
│  │         静止→重度平滑降噪 | 运动→快速响应降低延迟             │  │
│  └───────────────────────────────────────────────────────────┘  │
│                          │                                       │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              WebSocket /pose/stream (30fps 推送)            │  │
│  │              REST /pose/status + /pose/config               │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────┬───────────────────────────────────────┘
                          │ JSON (PoseFrame)
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Three.js 前端 (Vite + WebGL)                     │
│                                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │  WebSocket   │  │  Mecha       │  │  CameraPreview       │  │
│  │  客户端      │  │  Controller  │  │  (骨骼叠加层)        │  │
│  │  (自动重连)  │  │  (13段骨骼)  │  │  (280×210 Canvas)   │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
│                          │                                       │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  3D场景: 暗紫色背景 + 雾效 + 三点光照 + 阴影贴图 + 网格    │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 技术栈明细

| 层级 | 技术 | 版本 | 用途 |
|------|------|------|------|
| **后端框架** | FastAPI | ≥0.115.0 | HTTP + WebSocket 服务 |
| **ASGI 服务器** | Uvicorn | ≥0.30.0 | 异步服务运行 |
| **姿态检测** | MediaPipe | ≥0.10.0 | 33关键点全身姿态估计 |
| **目标检测** | YOLOv8n (Ultralytics) | ≥8.3.0 | 人物检测 + 姿态估计 |
| **深度学习** | PyTorch | ≥2.5.0 | YOLO 模型推理 |
| **图像处理** | OpenCV | ≥4.10.0 | 摄像头采集与预处理 |
| **数学计算** | NumPy | ≥2.0.0 | 坐标计算 |
| **数据校验** | Pydantic | ≥2.0.0 | 配置与数据模型 |
| **3D 渲染** | Three.js | ^0.170.0 | WebGL 实时渲染 |
| **构建工具** | Vite | ^6.0.0 | 前端开发与打包 |
| **运行环境** | Python 3.11+ / Node.js | - | - |

### 2.3 硬件环境

| 组件 | 型号 |
|------|------|
| GPU | NVIDIA GeForce RTX 5060 Ti 16GB (CUDA 加速) |
| CPU | AMD Ryzen 7 7700 |
| 内存 | 32GB DDR5 |
| 系统 | Windows 11 Pro |

---

## 三、已完成功能详解

### 3.1 三种姿态检测后端

系统实现了三种可运行时切换的姿态检测后端，通过工厂模式统一管理：

#### 3.1.1 MediaPipe 模式 (纯 MediaPipe)
- **文件**: `backend/app/services/pose_detection.py`
- **流程**: 摄像头 → BGR→RGB 转换 → MediaPipe PoseLandmarker → 33 关键点 → 取 13 个关键点子集 → PoseSmoother 滤波 → PoseFrame
- **特点**: Google 官方方案，精度高，33 个关键点含手部/足部细节
- **适用**: 对精度要求高的场景

#### 3.1.2 YOLO 模式 (纯 YOLOv8n-pose)
- **文件**: `backend/app/services/yolo_detection.py` (YoloPoseDetector + YoloPosePipeline)
- **流程**: 摄像头 → YOLOv8n-pose 端到端推理 → COCO 17 关键点 → 映射为 13 关键点 → Smoothing → PoseFrame
- **特点**: 端到端单模型，速度最快，适合资源受限场景
- **模型**: `backend/models/yolov8n-pose.pt` (约 5.5MB)

#### 3.1.3 混合模式 (Hybrid, 默认推荐)
- **文件**: `backend/app/services/yolo_detection.py` (HybridPosePipeline)
- **流程**: 摄像头 → YOLOv8n 人物检测 → 获得人物边界框 → 裁剪 ROI (15% margin) → MediaPipe 姿态估计 → 坐标映射回全帧 → Smoothing → PoseFrame
- **降级策略**: YOLO 未检测到人物或 ROI 太小(<32px)时自动降级为全帧 MediaPipe
- **优势**: 
  - ROI 裁剪减少 MediaPipe 输入尺寸 → 更快
  - 减少背景噪声干扰 → 更准
  - 双重保障 (YOLO 失败时降级) → 更稳

### 3.2 自适应时序滤波 (PoseSmoother)

- **文件**: `backend/app/services/pose_detection.py` (PoseSmoother 类)
- **核心算法**: 基于帧间位移速度的自适应低通滤波器
  - 静止时 (velocity≈0): alpha=min_alpha (0.25) → 重度平滑降噪
  - 运动时 (velocity 大): alpha 逼近 max_alpha (0.75) → 快速响应
  - 可见度修正: alpha *= visibility (低可见度时更依赖历史值)
- **效果**: 静止时消除抖动，运动时保持响应速度

### 3.3 实时 WebSocket 数据推送

- **文件**: `backend/app/api/routes/pose.py`
- **端点**: `ws://localhost:8000/pose/stream`
- **特点**:
  - 单例连接模式 (新连接踢旧连接)
  - 连接建立即时确认 (`{"type": "connected", "backend": "..."}`)
  - 目标 30fps 推送 (可配置)
  - 3 秒周期状态日志
  - 优雅降级与异常恢复

### 3.4 REST API 管理接口

| 方法 | 路径 | 功能 |
|------|------|------|
| GET | `/health` | 健康检查 |
| GET | `/docs` | Swagger UI (本地资源离线加载) |
| GET | `/pose/status` | 当前追踪状态 (FPS/延迟/检测状态/后端类型) |
| POST | `/pose/config` | 运行时切换检测后端 (`{"backend": "mediapipe|yolo|hybrid"}`) |

### 3.5 3D 机甲模型自动骨骼绑定

这是前端最复杂的模块 (`frontend/src/mecha.js`, 512 行):

#### 3.5.1 模型加载与分类
- **文件加载**: GLTFLoader 加载 GLB 格式机甲模型 (`public/机甲.glb`)
- **包围盒计算**: 自动计算模型空间包围盒，缩放至约 2.5 单位高
- **Mesh 自动分类**: 基于归一化坐标 (nx, ny, nz) 的空间区域分类器，将每个 mesh 分配到 13 个身体段:
  - `head` (ny ≥ 0.75)
  - `torso` (0.35 ≤ ny < 0.75, 中央区域)
  - `upperArmL/R` (0.55 ≤ ny < 0.75, 左右侧)
  - `lowerArmL/R` (0.35 ≤ ny < 0.55, 左右侧)
  - `hip` (0.22 ≤ ny < 0.35)
  - `upperLegL/R` (0.10 ≤ ny < 0.22)
  - `lowerLegL/R` (0.03 ≤ ny < 0.10)
  - `footL/R` (ny < 0.03)
- **缺段处理**: 对于无 mesh 的身体段，通过相邻段推测位置 (`_estimateCenter`)
- **诊断输出**: 详细的 Y 高度带分布、段分配结果、段中心坐标日志

#### 3.5.2 层次化骨骼结构
- 使用 `THREE.Group` 构建 13 段父子层级:
  - `hip` → `upperLegL/R` → `lowerLegL/R` → `footL/R`
  - `torso` → `head`, `upperArmL/R` → `lowerArmL/R`
- 每个段的 pivot 设置为其关节位置
- Mesh 从 GLTF 场景树迁移到对应段 Group，保留世界空间变换

#### 3.5.3 实时姿态驱动
- **坐标转换**: `mpToScene()` 将归一化 MediaPipe 坐标映射到 Three.js 场景坐标
- **方向向量计算**: 从关键点对计算每个身体段的方向向量 (如肩→肘、髋→膝等)
- **四元数旋转**: `Quaternion.setFromUnitVectors()` 从 rest 方向旋转到当前方向
- **SLERP 平滑**: 使用球面线性插值 (alpha=0.55) 平滑旋转过渡
- **位移跟随**: 整个模型跟随髋部中点 (alpha=0.65)

### 3.6 姿态可视化叠加层

- **文件**: `frontend/src/cameraPreview.js`
- **功能**: 280×210 Canvas 上的骨骼叠加层
- **显示内容**:
  - 13 关键点的彩色标记 (不同身体区域不同颜色)
  - 12 条骨骼连线 (鼻→肩→肘→腕, 肩→髋→膝→踝)
  - 可见度过滤 (visibility < 0.3 时隐藏)
- **运行模式**: 叠加层模式 (后端独占摄像头，前端仅渲染骨骼)

### 3.7 3D 场景系统

- **文件**: `frontend/src/scene.js`
- **渲染器**: WebGLRenderer (抗锯齿, PCFSoft 阴影, 高性能模式)
- **场景**: 暗紫色背景 (0x1a1a2e) + 距离雾效
- **相机**: 透视相机 (FOV 60°, 位置 (0, 1.5, 3.5), 注视 (0, 0.8, 0))
- **光照**: 三点光照系统
  - 环境光 (蓝调, 强度 1.5)
  - 主方向光 (白色, 强度 2.0, 阴影投射)
  - 补光 (蓝色调, 强度 0.8)
  - 边缘光 (白色, 强度 0.6, 背光)
- **地面**: 20×20 网格参考面

### 3.8 WebSocket 客户端 (前端)

- **文件**: `frontend/src/websocket.js`
- **核心特性**:
  - 自动重连 (指数退避: 1.5s → 2.25s → ... → 最大 15s)
  - 连接状态 DOM 更新 (已连接/已断开 + 后端类型标签)
  - 系统消息识别 (`"type": "connected"`)
  - 后端类型动态展示 ("MediaPipe" / "YOLO" / "混合模式")
  - 3 秒无数据看门狗检测
  - 日志节流 (前 5 条全量, 之后每 30 条一次)

### 3.9 数据采集与训练流水线

#### 3.9.1 数据采集工具
- **文件**: `backend/scripts/collect_training_data.py`
- **功能**: 打开摄像头运行 MediaPipe，按 `S` 键保存标注帧
- **输出**: YOLO 格式 (JPEG 图片 + 边界框 + 17 COCO 关键点标签)
- **特点**: 实时叠加可视化 (关键点+骨架+边界框+FPS+计数)

#### 3.9.2 训练脚本
- **文件**: `backend/scripts/train_yolo_pose.py`
- **功能**: 使用 MediaPipe 自动标注数据进行知识蒸馏训练
- **配置**: `backend/configs/yolo_train.yaml`
- **训练参数**: 100 epochs, imgsz=640, batch=16, lr=0.001, 早停 patience=20
- **数据增强**: mosaic, mixup, copy-paste, HSV 抖动, 旋转, 缩放, 翻转
- **输出**: `backend/models/yolo_pose_custom.pt` + ONNX 导出

### 3.10 性能基准测试工具

- **文件**: `backend/scripts/benchmark.py`
- **功能**: 对比三种后端在相同视频/摄像头输入下的表现
- **指标**: 总帧数, 检测帧数, 检测率(%), 平均延迟(ms), 平均FPS, 关键点抖动
- **输出**: 格式化对比表格

### 3.11 配置管理系统

- **文件**: `backend/app/core/config.py`
- **方案**: Pydantic Settings 从 `.env` 文件自动加载
- **配置项** (18 项):
  - Server: host, port
  - Camera: index, width, height, fps
  - MediaPipe: model_complexity, min_detection_confidence, min_tracking_confidence
  - YOLO: backend, person_model, pose_model, confidence, iou
  - Assets: model_dir

### 3.12 数据模型设计

- **文件**: `backend/app/models/pose.py`
- **核心类型**:
  - `Landmark`: 单关键点 (x, y, z, visibility)
  - `PoseFrame`: 完整帧 (timestamp, 13 landmarks, world_landmarks, backend)
  - `BackendType`: 枚举 (mediapipe, yolo, hybrid)
  - `CameraConfig`, `TrackingStatus`
- **关键点映射**:
  - `BASIC_LANDMARK_INDICES`: MediaPipe 33 → 项目 13 关键点的索引映射
  - `COCO_TO_BASIC`: COCO 17 → 项目 13 关键点的索引映射

---

## 四、项目文件结构

```
my_digital_twin/
├── README.md                           # 项目说明
├── start.bat / start.sh                # 一键启动脚本
├── docs/
│   ├── requirements.md                 # 一阶段：需求分析
│   ├── tech_selection.md               # 二阶段：技术选型
│   └── yolo_integration_plan.md        # YOLOv8 集成方案
├── backend/
│   ├── .env                            # 环境配置
│   ├── requirements.txt                # Python 依赖
│   ├── setup.py                        # 资源下载脚本
│   ├── configs/
│   │   └── yolo_train.yaml            # YOLO 训练超参数
│   ├── models/
│   │   ├── pose_landmarker_lite.task   # MediaPipe 模型
│   │   ├── yolov8n.pt                  # YOLOv8n 人物检测
│   │   └── yolov8n-pose.pt            # YOLOv8n-pose 姿态估计
│   ├── scripts/
│   │   ├── collect_training_data.py    # 数据采集工具
│   │   ├── train_yolo_pose.py          # 训练脚本
│   │   └── benchmark.py               # 性能基准测试
│   └── app/
│       ├── main.py                     # FastAPI 入口
│       ├── core/
│       │   └── config.py              # 配置管理
│       ├── models/
│       │   └── pose.py                # 数据模型
│       ├── api/routes/
│       │   └── pose.py                # WebSocket + REST API
│       └── services/
│           ├── pose_detection.py       # MediaPipe 服务
│           └── yolo_detection.py       # YOLO + Hybrid 服务
└── frontend/
    ├── index.html                      # SPA 入口
    ├── package.json                    # 依赖配置
    ├── vite.config.js                  # Vite 配置
    ├── public/
    │   └── 机甲.glb                    # 3D 机甲模型
    └── src/
        ├── main.js                     # 应用入口
        ├── scene.js                    # 3D 场景
        ├── websocket.js                # WebSocket 客户端
        ├── cameraPreview.js            # 骨骼叠加层
        └── mecha.js                    # 机甲控制器
```

---

## 五、Git 提交历史

| 提交 | 说明 |
|------|------|
| `97088df` | Initial commit |
| `e236628` | feat: 初始化项目 |
| `e676b51` | feat: 集成YOLOv8人物检测与混合姿态估计流水线 |
| `8824eb6` | feat: 增加yolov8n模型 |
| `106e17d` | fix(frontend): 优化前端模型计算 |

---

## 六、关键技术决策

1. **后端独占摄像头**: 后端 OpenCV 直接采集摄像头，前端仅渲染骨骼叠加层，避免浏览器摄像头权限问题
2. **13 关键点子集**: 从 MediaPipe 的 33 点和 COCO 的 17 点中选取 13 个核心关节点，聚焦身体动作驱动
3. **自适应平滑**: 基于运动速度的动态滤波系数，平衡降噪与响应
4. **混合流水线**: YOLO 人物检测 + 裁剪 ROI + MediaPipe，兼顾速度与精度
5. **单例 WebSocket**: 每次只允许一个客户端连接，新连接踢旧连接
6. **运行时后端切换**: 无需重启服务即可在三种后端间切换，便于 A/B 对比
7. **GLB 格式**: 选择 glTF Binary 作为 3D 模型格式，加载效率高且广泛支持
8. **空间区域分类**: 基于归一化坐标的 mesh 自动分类，无需手动标注骨骼绑定
