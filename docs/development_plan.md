# My Digital Twin 后续发展规划与架构设计

> 日期: 2026-05-24 | 版本: 0.2.0-dev

---

## 一、发展路线图

### 阶段概览

```
当前 v0.1.0          v0.2.0              v0.3.0              v1.0.0
    │                   │                   │                   │
    ▼                   ▼                   ▼                   ▼
┌─────────┐      ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ 基础    │      │ 功能完善    │     │ 生产化      │     │ 正式发布    │
│ 原型    │ ──►  │ 与优化      │ ──► │ 与跨平台    │ ──► │ 与分发      │
└─────────┘      └─────────────┘     └─────────────┘     └─────────────┘
  ✅ 已完成          🔜 下一步         📋 计划中          🎯 最终目标
```

---

### v0.2.0 — 功能完善与性能优化 (预计 4-6 周)

#### 1. 自定义训练模型部署
- [ ] 采集 500+ 帧训练数据 (多角度、多光照、多姿态)
- [ ] 训练 YOLOv8n-pose 自定义模型 (知识蒸馏)
- [ ] 与 MediaPipe 进行精度对比评估 (PCK/OKS 指标)
- [ ] 验证自定义模型在混合流水线中的表现

#### 2. 3D 模型系统增强
- [ ] 模型热切换: 支持运行时更换 3D 模型 (如切换不同机甲/角色)
- [ ] 模型缓存: 预加载常用模型，减少切换延迟
- [ ] 骨架重定向: 将姿态数据适配到不同比例的人形模型
- [ ] 手部姿态: 集成 MediaPipe Hands，增加手指关节驱动

#### 3. 动画系统升级
- [ ] IK (反向运动学): 解决脚部穿地、手腕位置偏差问题
- [ ] 动作平滑: 动画状态机 + 动作过渡 (避免突变)
- [ ] 物理模拟: 头发/布料等附属物的简单物理效果
- [ ] 面部表情: 集成 MediaPipe Face Mesh (478 点面部网格)

#### 4. 场景与交互
- [ ] 场景切换: 多场景支持 (舞台/户外/科幻场景)
- [ ] 虚拟摄像头: 自由旋转/缩放视角 (OrbitControls)
- [ ] UI 面板: 显示 FPS、延迟、后端状态、模型名称
- [ ] 快捷键: 切换后端/模型/场景

---

### v0.3.0 — 生产化与跨平台 (预计 6-8 周)

#### 1. Unity 前端替换 (核心变更)

当前 Three.js 前端为开发原型，需要替换为 Unity 以支持生产级特性：

```
当前架构 (v0.1.0):                    目标架构 (v0.3.0+):
                                      
  Three.js (浏览器)                    Unity (原生应用)
  ┌───────────────┐                   ┌────────────────────┐
  │ WebGL 渲染    │                   │ DirectX 12/Vulkan  │
  │ 单窗口        │        ──►        │ 60fps 原生渲染     │
  │ 无原生摄像头  │                   │ WebCamTexture      │
  │ 有限后处理    │                   │ 完整后处理管线     │
  └───────────────┘                   └────────────────────┘
```

- [ ] Unity 项目搭建 (2022.3 LTS)
- [ ] WebSocket 客户端 (NativeWebSocket / UnityWebSocket)
- [ ] 3D 模型导入与 Humanoid Rig 配置
- [ ] Animator + Humanoid Avatar 骨骼映射
- [ ] 场景系统 (光照/阴影/后处理)
- [ ] Windows 平台导出 (.exe)

#### 2. Unity 端摄像头直连模式

```
方案A (当前): 后端 OpenCV → WebSocket → Unity
方案B (新增): Unity WebCamTexture → 本地推理 → 渲染 (无网络延迟)

以方案B为主，方案A作为远程/多人观察模式保留
```

- [ ] ONNX Runtime / Barracuda 集成 YOLO 模型
- [ ] 或通过本地 HTTP 连接同机 FastAPI 后端
- [ ] 摄像头权限管理

#### 3. Android 平台适配
- [ ] Unity Android Build 配置
- [ ] 移动端性能优化 (模型量化 INT8, 降低分辨率)
- [ ] 触屏交互适配
- [ ] APK 打包与签名

#### 4. 多人姿态追踪
- [ ] 后端支持多实例流水线
- [ ] 多人关键点 ID 追踪 (ByteTrack/BoT-SORT)
- [ ] 多机甲同时渲染
- [ ] WebSocket 频道隔离 (每用户独立频道)

---

### v1.0.0 — 正式发布 (预计 4-6 周)

- [ ] 性能压力测试与优化
- [ ] 安装包制作 (Windows .exe + Android .apk)
- [ ] 用户文档与使用指南
- [ ] 自动更新机制
- [ ] 错误上报与诊断

---

## 二、Unity 前端详细架构

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                     Unity Application                            │
│                                                                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    Scene Manager                           │  │
│  │   ┌──────────┐  ┌──────────┐  ┌──────────────────────┐   │  │
│  │   │ 主场景   │  │ UI场景   │  │ 加载/过渡场景        │   │  │
│  │   └──────────┘  └──────────┘  └──────────────────────┘   │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                   Pose Manager                             │  │
│  │                                                             │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │  │
│  │  │ 本地推理模式 │  │ 远程WebSocket│  │ 录制/回放    │    │  │
│  │  │ (ONNX/本地) │  │ (连接后端)   │  │ (离线数据)   │    │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘    │  │
│  └───────────────────────────────────────────────────────────┘  │
│                          │                                       │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                   Avatar Controller                        │  │
│  │                                                             │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │  │
│  │  │ IK Solver    │  │ Bone Mapper  │  │ Animation    │    │  │
│  │  │ (脚/手IK)   │  │ (骨骼映射)   │  │ Blender      │    │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘    │  │
│  └───────────────────────────────────────────────────────────┘  │
│                          │                                       │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                   3D Rendering                              │  │
│  │                                                             │  │
│  │  URP/HDRP Pipeline → Post Processing → Output (60fps)      │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 核心模块设计

#### 2.2.1 PoseManager (姿态管理器)

```csharp
// 姿态数据来源抽象
public interface IPoseSource
{
    event Action<PoseFrame> OnPoseReceived;
    bool IsActive { get; }
    void Start();
    void Stop();
}

// 本地推理实现 (ONNX Runtime 或本地 HTTP)
public class LocalPoseSource : IPoseSource { ... }

// WebSocket 远程实现
public class RemotePoseSource : IPoseSource { ... }

// PoseManager 统一管理
public class PoseManager : MonoBehaviour
{
    public IPoseSource ActiveSource { get; private set; }
    
    public void SwitchSource(PoseSourceType type);
    public PoseFrame GetLatestPose();
    
    // 录制/回放
    public void StartRecording(string path);
    public void StopRecording();
    public void PlayRecording(string path);
}
```

#### 2.2.2 AvatarController (虚拟人控制器)

```csharp
public class AvatarController : MonoBehaviour
{
    [Header("References")]
    public Animator animator;           // Humanoid Animator
    public Transform hipBone;           // 根骨骼引用
    public Transform[] targetBones;     // 13 个目标骨骼
    
    [Header("Settings")]
    public float smoothSpeed = 12f;     // 平滑速度
    public bool useIK = true;           // 启用 IK
    
    // 核心方法
    public void ApplyPose(PoseFrame pose);
    
    // IK 修正
    private void SolveFootIK();
    private void SolveHandIK();
    private void UpdateBoneRotations();
    private void UpdateRootPosition();
}
```

#### 2.2.3 BoneMapper (骨骼映射器)

```csharp
// Humanoid Avatar 骨骼到 13 关键点的映射配置
[CreateAssetMenu(fileName = "BoneMapping", menuName = "DT/Bone Mapping")]
public class BoneMappingConfig : ScriptableObject
{
    public HumanBodyBones[] keypointToBone;  // [13] 映射表
    
    // keypoint[0]=Nose → HumanBodyBones.Head
    // keypoint[1]=LShoulder → HumanBodyBones.LeftUpperArm
    // keypoint[3]=LElbow → HumanBodyBones.LeftLowerArm
    // keypoint[5]=LWrist → HumanBodyBones.LeftHand
    // ...
}
```

### 2.3 渲染管线

| 特性 | Windows | Android |
|------|---------|---------|
| 渲染管线 | URP (Universal) | URP (Universal) |
| 图形 API | DirectX 12 | Vulkan |
| 阴影 | 实时阴影 1024² | 静态烘焙 |
| 后处理 | Bloom + AO + Color Grading | 简化 (仅 Bloom) |
| 抗锯齿 | TAA / SMAA | FXAA |
| 目标帧率 | 60fps | 30fps |

### 2.4 Unity 项目结构 (规划)

```
Assets/
├── _Project/
│   ├── Scenes/
│   │   ├── MainScene.unity
│   │   ├── LoadingScene.unity
│   │   └── UIScene.unity
│   ├── Scripts/
│   │   ├── Core/
│   │   │   ├── PoseManager.cs
│   │   │   ├── AvatarController.cs
│   │   │   └── BoneMapper.cs
│   │   ├── Network/
│   │   │   ├── WebSocketClient.cs
│   │   │   └── PoseMessage.cs
│   │   ├── IK/
│   │   │   ├── FootIKSolver.cs
│   │   │   └── HandIKSolver.cs
│   │   ├── UI/
│   │   │   ├── DebugPanel.cs
│   │   │   ├── ModelSelector.cs
│   │   │   └── SettingsPanel.cs
│   │   └── Utils/
│   │       ├── LandmarkConverter.cs
│   │       └── PoseRecorder.cs
│   ├── Models/
│   │   ├── Mecha/          # 机甲模型
│   │   └── Characters/     # 其他角色模型
│   ├── Animations/
│   │   └── Controllers/    # Animator Controllers
│   ├── Materials/
│   ├── Prefabs/
│   └── ScriptableObjects/
│       └── BoneMappingConfig.asset
├── Plugins/
│   ├── NativeWebSocket/
│   └── ONNXRuntime/
└── Resources/
    └── Models/
        └── yolo_n_custom.onnx
```

---

## 三、多人架构设计

### 3.1 多人模式数据流

```
┌────────────┐   ┌────────────┐   ┌────────────┐
│  用户 A    │   │  用户 B    │   │  观察者 C  │
│ (摄像头1)  │   │ (摄像头2)  │   │ (无摄像头) │
└─────┬──────┘   └─────┬──────┘   └─────┬──────┘
      │                │                │
      ▼                ▼                │
┌─────────────────────────────────────────────────┐
│                FastAPI 后端                       │
│                                                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │
│  │Pipeline 1│  │Pipeline 2│  │频道管理器    │   │
│  │(user_a)  │  │(user_b)  │  │(广播/单播)  │   │
│  └──────────┘  └──────────┘  └──────────────┘   │
│                                                   │
│  WebSocket 频道:                                  │
│    /pose/stream/user_a  → 仅用户A的姿态          │
│    /pose/stream/user_b  → 仅用户B的姿态          │
│    /pose/stream/room_1  → 房间内所有人的姿态     │
└─────────────────────────────────────────────────┘
      │                │                │
      ▼                ▼                ▼
┌─────────────────────────────────────────────────┐
│              Unity 客户端                         │
│  渲染多个虚拟人 (每人一个 AvatarController)      │
└─────────────────────────────────────────────────┘
```

### 3.2 后端多人改造

```python
# backend/app/core/multi_tracker.py (新增)
class MultiPersonTracker:
    """多人追踪管理器"""
    
    def __init__(self):
        self.pipelines: dict[str, Pipeline] = {}
        self.rooms: dict[str, set[str]] = {}
    
    async def add_user(self, user_id: str, camera_index: int):
        pipeline = HybridPosePipeline(camera_index=camera_index)
        self.pipelines[user_id] = pipeline
    
    async def broadcast_room(self, room_id: str):
        """向房间内所有用户广播姿态数据"""
        ...
    
    async def remove_user(self, user_id: str):
        pipeline = self.pipelines.pop(user_id, None)
        if pipeline:
            pipeline.close()

# WebSocket 端点新增
@router.websocket("/stream/{user_id}")          # 单用户流
@router.websocket("/room/{room_id}/{user_id}")   # 房间多用户流
```

---

## 四、后端服务化架构

### 4.1 当前单体 → 微服务拆分 (远期)

```
当前 (单体):
  FastAPI 进程 = Web服务 + 摄像头 + 推理 + 推送

远期 (微服务, 可选):
  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
  │ Gateway      │   │ Pose Worker  │   │ Stream Hub   │
  │ (Nginx/     │   │ (推理进程)   │   │ (推送中心)   │
  │  Traefik)   │   │ x N          │   │              │
  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘
         │                  │                  │
         └──────────────────┼──────────────────┘
                            │
                    ┌───────┴───────┐
                    │ Redis / NATS  │
                    │ (消息总线)    │
                    └───────────────┘
```

### 4.2 模型版本管理

```
backend/models/
├── pose_landmarker_lite.task     # MediaPipe Lite (当前)
├── pose_landmarker_full.task     # MediaPipe Full (高精度场景)
├── pose_landmarker_heavy.task    # MediaPipe Heavy (超高精度)
├── yolov8n.pt                    # YOLOv8n 人物检测 (当前)
├── yolov8s.pt                    # YOLOv8s 人物检测 (均衡)
├── yolov8n-pose.pt               # YOLOv8n-pose 姿态 (当前)
├── yolov8s-pose.pt               # YOLOv8s-pose 姿态 (高精度)
├── yolo_pose_custom.pt           # 自定义蒸馏模型
└── model_registry.json           # 模型注册表
```

### 4.3 配置中心化

```json
// model_registry.json (模型注册表)
{
  "models": {
    "person_yolo_n": {
      "path": "yolov8n.pt",
      "type": "detect",
      "size_mb": 5.5,
      "device": "cuda",
      "precision": "fp16"
    },
    "pose_mp_lite": {
      "path": "pose_landmarker_lite.task",
      "type": "pose",
      "backend": "mediapipe",
      "device": "cpu"
    }
  },
  "profiles": {
    "performance": {
      "person": "person_yolo_n",
      "pose": "pose_mp_lite",
      "resolution": "480p"
    },
    "quality": {
      "person": "person_yolo_s",
      "pose": "pose_mp_full",
      "resolution": "720p"
    }
  }
}
```

---

## 五、性能优化路线

### 5.1 推理层优化

| 优化项 | 方法 | 预期收益 |
|--------|------|----------|
| 模型量化 | INT8/FP16 量化 YOLO 模型 | 2-3x 推理加速 |
| TensorRT 部署 | YOLO → TensorRT Engine | 3-5x 推理加速 |
| 帧跳过 | 每 2 帧检测一次 (中间帧复用) | 50% 计算减少 |
| 分辨率降采样 | 推理时使用 480p, 渲染用 720p | 30% 加速 |
| 批处理 | 合并多人推理为 batch | GPU 利用率提升 |

### 5.2 传输层优化

| 优化项 | 方法 | 预期收益 |
|--------|------|----------|
| 二进制协议 | JSON → MessagePack/Protobuf | 50-70% 带宽减少 |
| 关键帧机制 | 全量帧 + 增量帧交替 | 60% 数据量减少 |
| 压缩 | WebSocket Per-Message Deflate | 50-80% 带宽减少 |

### 5.3 渲染层优化

| 优化项 | 方法 | 预期收益 |
|--------|------|----------|
| LOD | 多细节层次模型 | 远处模型减少面数 |
| GPU Instancing | 多人渲染合批 | Draw Call 减少 |
| Occlusion Culling | 视锥体剔除 | 不可见模型不渲染 |
| Shader 简化 | Android 端使用 Unlit/SimpleLit | 移动端流畅 |

---

## 六、项目关键风险与对策

| 风险 | 影响 | 对策 |
|------|------|------|
| Unity 骨骼映射不准确 | 动画效果差 | 提供手动校准工具 + BoneMappingConfig 可配置 |
| Android 推理性能不足 | 卡顿 | ONNX 量化 + 分辨率降级 + 帧跳过 |
| 暗光环境检测失败 | 用户无法使用 | 增加图像预处理 (直方图均衡化) + 红外摄像头支持 |
| 多人模式下带宽爆炸 | 延迟增高 | 二进制协议 + 增量帧 + 自适应码率 |
| 模型授权问题 | 分发受阻 | 使用自有/开源模型，避免商业模型依赖 |

---

## 七、技术债务清理计划

| 项目 | 当前状态 | 目标 |
|------|----------|------|
| 单元测试 | 无 | backend 覆盖率 ≥ 60% |
| 类型标注 | 基本完成 | 100% mypy strict 通过 |
| 错误处理 | 基础 try/except | 统一错误码 + 告警 |
| 日志系统 | print() | 结构化日志 (structlog/loguru) |
| API 版本化 | 无 | `/api/v1/pose/stream` |
| CI/CD | 无 | GitHub Actions (lint + test) |
| 配置校验 | 无 | Pydantic validator 校验路径/范围 |
