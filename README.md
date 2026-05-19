# my_digital_twin

实时人体姿态驱动的 3D 数字孪生游戏软件。通过摄像头实时捕捉人体动作，将姿态数据映射到 3D 模型上，实现真人与虚拟角色的实时同步。

本项目已集成 [andrej-karpathy-skills](https://github.com/forrestchang/andrej-karpathy-skills) 的 Cursor 行为准则。规则文件位于 [`.cursor/rules/karpathy-guidelines.mdc`](.cursor/rules/karpathy-guidelines.mdc)。

## 项目结构

```
my_digital_twin/
├── docs/
│   ├── requirements.md        # 一阶段：需求分析
│   └── tech_selection.md      # 二阶段：技术选型
├── backend/                   # FastAPI 后端
│   ├── app/
│   │   ├── main.py            # 服务入口
│   │   ├── api/routes/        # API 路由
│   │   │   └── pose.py        # 姿态数据 WebSocket + REST
│   │   ├── core/
│   │   │   └── config.py      # 配置管理
│   │   ├── models/
│   │   │   └── pose.py        # 姿态数据模型
│   │   └── services/
│   │       └── pose_detection.py  # MediaPipe 姿态检测
│   ├── requirements.txt
│   └── .env
├── frontend/                  # Three.js 开发前端
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   └── src/
│       ├── main.js            # 入口
│       ├── scene.js           # 3D 场景 + 骨骼可视化
│       └── websocket.js       # WebSocket 客户端
├── assets/                    # 3D 模型、贴图等资源
│   └── models/                # Blender 导出的 glTF/FBX 模型
└── README.md
```

## 技术栈

| 层 | 技术 |
|----|------|
| 后端框架 | FastAPI (Python 3.11+) |
| 姿态检测 | MediaPipe Pose |
| 实时通信 | WebSocket |
| 开发前端 | Three.js + Vite |
| 生产前端 | Unity (2022.3 LTS) |
| 平台 | Windows 11 + Android |

## 快速启动

### 一键启动 (推荐)

```bash
# Windows - 双击运行
start.bat

# Git Bash / WSL
bash start.sh
```

脚本会自动安装依赖并同时启动后端和前端。

### 手动启动

**后端** (`http://localhost:8000`，API 文档 `http://localhost:8000/docs`):

```bash
cd backend
pip install -r requirements.txt
python -m app.main
```

**前端** (`http://localhost:5173`):

```bash
cd frontend
npm install
npm run dev
```

## 核心数据流

```
摄像头 → MediaPipe 姿态检测 → PoseFrame (33 个 3D 关键点)
                ↕
        FastAPI WebSocket
                ↕
    Three.js / Unity 3D 渲染
```

## 路线图

- [x] 一阶段：需求分析
- [x] 二阶段：技术选型
- [x] 基础项目框架搭建
- [ ] 后端姿态检测调优
- [ ] Unity 前端集成
- [ ] 3D 模型骨骼映射
- [ ] Android 平台适配
