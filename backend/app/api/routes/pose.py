from __future__ import annotations

import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from app.core.config import settings
from app.models.pose import TrackingStatus

router = APIRouter(prefix="/pose", tags=["pose"])

# 全局流水线实例
_pipeline: object | None = None
_pipeline_backend: str = ""
_fps_buffer: list[float] = []
_last_frame_time: float = 0.0
_detected: bool = False
_active_ws: WebSocket | None = None  # 当前活跃的 WebSocket, 新连接到达时踢掉旧连接


class BackendConfig(BaseModel):
    backend: str = "hybrid"  # "mediapipe" | "yolo" | "hybrid"


def _get_pipeline():
    """根据配置创建/复用流水线"""
    global _pipeline, _pipeline_backend

    target = settings.pose_backend
    if _pipeline is not None and _pipeline_backend == target:
        return _pipeline

    # 关闭旧流水线
    if _pipeline is not None:
        try:
            _pipeline.close()
        except Exception:
            pass

    if target == "mediapipe":
        from app.services.pose_detection import PosePipeline
        _pipeline = PosePipeline()
    elif target == "yolo":
        from app.services.yolo_detection import YoloPosePipeline
        _pipeline = YoloPosePipeline()
    elif target == "hybrid":
        from app.services.yolo_detection import HybridPosePipeline
        _pipeline = HybridPosePipeline()
    else:
        raise ValueError(f"Unknown pose backend: {target}")

    _pipeline_backend = target
    return _pipeline


def _update_stats(has_detection: bool) -> None:
    global _fps_buffer, _last_frame_time, _detected
    _detected = has_detection
    now = time.time()
    if _last_frame_time > 0:
        dt = now - _last_frame_time
        if dt > 0:
            _fps_buffer.append(1.0 / dt)
            if len(_fps_buffer) > 60:
                _fps_buffer = _fps_buffer[-60:]
    _last_frame_time = now


@router.websocket("/stream")
async def pose_stream(websocket: WebSocket) -> None:
    """WebSocket: 实时推送姿态数据"""
    global _active_ws, _pipeline, _pipeline_backend

    # 如果有旧连接 (页面刷新等), 先踢掉
    if _active_ws is not None:
        print("[WS] 踢掉旧连接, 接受新连接")
        try:
            await _active_ws.close(code=1001, reason="新连接取代")
        except Exception:
            pass
        _active_ws = None

    await websocket.accept()
    _active_ws = websocket
    print(f"[WS] 客户端已连接, 启动流水线 (backend={settings.pose_backend})...", flush=True)
    pl = _get_pipeline()
    print(f"[WS] 流水线已创建: {type(pl).__name__}", flush=True)

    # 发送即时确认消息, 让前端知道通道已建立
    await websocket.send_json({"type": "connected", "backend": _pipeline_backend})

    ws_frame_count = 0
    try:
        async for pose_frame in pl.stream():
            _update_stats(True)
            ws_frame_count += 1
            if ws_frame_count <= 3 or ws_frame_count % 30 == 0:
                print(f"[WS] 发送帧 #{ws_frame_count} backend={pose_frame.backend} "
                      f"landmarks={len(pose_frame.landmarks)}", flush=True)
            try:
                await websocket.send_json(pose_frame.model_dump())
            except Exception:
                break
    except WebSocketDisconnect:
        print(f"[WS] 客户端断开, 共发送{ws_frame_count}帧", flush=True)
    except Exception as e:
        print(f"[WS] 流异常: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("[WS] 连接已断开, 清理流水线", flush=True)
        if _active_ws is websocket:
            _active_ws = None
        if _pipeline is not None:
            try:
                _pipeline.close()
            except Exception:
                pass
            _pipeline = None
            _pipeline_backend = ""


@router.get("/status", response_model=TrackingStatus)
async def get_status() -> TrackingStatus:
    """获取当前追踪状态"""
    avg_fps = sum(_fps_buffer) / len(_fps_buffer) if _fps_buffer else 0.0
    return TrackingStatus(
        active=_pipeline is not None,
        fps=round(avg_fps, 1),
        latency_ms=round(1000.0 / avg_fps, 1) if avg_fps > 0 else 0.0,
        detected=_detected,
        backend=_pipeline_backend or settings.pose_backend,
    )


@router.post("/config")
async def set_backend(config: BackendConfig) -> dict:
    """运行时切换检测后端"""
    global _pipeline, _pipeline_backend

    valid = {"mediapipe", "yolo", "hybrid"}
    if config.backend not in valid:
        return {"error": f"Invalid backend. Choose from: {valid}"}

    settings.pose_backend = config.backend

    # 关闭旧流水线
    if _pipeline is not None:
        try:
            _pipeline.close()
        except Exception:
            pass
        _pipeline = None
        _pipeline_backend = ""

    return {"status": "ok", "backend": config.backend}
