# 二阶段：技术选型

## 1. 硬件环境

| 组件 | 型号 |
|------|------|
| GPU | NVIDIA GeForce RTX 5060 Ti 16GB |
| CPU | AMD Ryzen 7 7700 |
| 内存 | 32GB DDR5 |
| 系统 | Windows 11 专业版 |

GPU 支持 CUDA、TensorRT、DirectML 等加速方案，对 AI 推理和 3D 渲染都有强力支撑。

## 2. 后端技术栈

### 2.1 Web 框架：FastAPI (Python 3.11+)

| 维度 | 选择理由 |
|------|----------|
| **实时通信** | 原生 WebSocket 支持，适合姿态数据流 |
| **性能** | 异步非阻塞，基于 Starlette + Uvicorn |
| **AI 集成** | 原生 Python 生态，无缝对接 MediaPipe/OpenCV/numpy |
| **开发效率** | 自动 API 文档 (Swagger)、类型提示 |
| **部署** | 独立进程，局域网通信，延迟低 |

### 2.2 姿态估计引擎：MediaPipe Pose

| 维度 | 选择理由 |
|------|----------|
| **精度** | 33 个 3D 关键点，含手部/足部细节 |
| **性能** | GPU 加速（CUDA/DirectML），RTX 5060 Ti 上 < 10ms/帧 |
| **跨平台** | Windows + Android 均支持 |
| **社区** | Google 维护，成熟稳定 |

备选方案：OpenCV + OpenPose（精度更高但性能开销大，作为可替换方案保留接口）。

### 2.3 其他后端依赖

| 库 | 用途 |
|----|------|
| `opencv-python` | 摄像头帧捕获与预处理 |
| `mediapipe` | 姿态关键点检测 |
| `numpy` | 关键点坐标计算 |
| `fastapi` + `uvicorn` | Web 服务框架 |
| `websockets` | WebSocket 通信 |
| `pydantic` | 数据校验与模型定义 |

## 3. 前端技术栈

### 3.1 推荐方案：Unity (C#)

| 维度 | 选择理由 |
|------|----------|
| **3D 渲染** | 原生高性能 3D 引擎，支持 Blender 导出的 FBX/glTF 模型 |
| **骨骼动画** | Animator + Humanoid Rig 完美支持姿态数据驱动的骨骼动画 |
| **跨平台** | 一次开发，导出 Windows (.exe) + Android (.apk) |
| **GPU 加速** | DirectX 12 (Windows) / Vulkan (Android)，充分利用 RTX 5060 Ti |
| **摄像头** | Unity WebCamTexture 原生支持摄像头采集 |
| **网络** | UnityWebSocket / NativeWebSocket 对接 FastAPI |
| **生态** | Asset Store 丰富，社区成熟 |

### 3.2 备选方案：Three.js + React

| 优点 | 缺点 |
|------|------|
| Web 技术栈，开发快 | 3D 性能不如原生引擎 |
| WebSocket 原生支持 | Android 需 WebView 封装 |
| 易于调试 | 大型模型加载效率低 |

在当前需求（游戏级 3D 实时渲染 + Android 支持）下，**Unity 明显优于 Web 方案**。

## 4. 通信协议

| 层 | 协议 | 说明 |
|----|------|------|
| 实时姿态数据 | **WebSocket** | 双向低延迟，JSON/二进制帧 |
| 模型/资源配置 | **HTTP REST** | 模型列表、配置下发 |
| 视频帧传输 | **raw bytes / base64** | 可压缩为 JPEG 降低带宽 |

## 5. 数据流设计

```
Unity (摄像头采集)
    │  RGB 帧
    ▼
FastAPI (WebSocket / HTTP)
    │  MediaPipe 推理
    ▼
Pose Landmarks (33点 × 3D 坐标)
    │  JSON / Protobuf
    ▼
Unity (Animator 驱动)
    │  骨骼映射 → 3D 模型渲染
    ▼
画面输出 (60fps)
```

## 6. 开发工具

| 工具 | 用途 |
|------|------|
| VS Code / Cursor | 后端开发 |
| Unity 2022.3 LTS | 前端开发 |
| Git | 版本控制 |
| Blender | 3D 模型制作与导出 |
