from __future__ import annotations

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.models.pose import PoseFrame, TrackingStatus
from app.services.pose_detection import PosePipeline

router = APIRouter(prefix="/pose", tags=["pose"])

pipeline: PosePipeline | None = None


def get_pipeline() -> PosePipeline:
    global pipeline
    if pipeline is None:
        pipeline = PosePipeline()
    return pipeline


@router.websocket("/stream")
async def pose_stream(websocket: WebSocket) -> None:
    """WebSocket 端点：实时推送姿态数据"""
    await websocket.accept()
    pl = get_pipeline()

    try:
        async for pose_frame in pl.stream():
            await websocket.send_json(pose_frame.model_dump())
    except WebSocketDisconnect:
        pass
    except Exception:
        pass


@router.get("/status", response_model=TrackingStatus)
async def get_status() -> TrackingStatus:
    """获取当前追踪状态"""
    return TrackingStatus(
        active=pipeline is not None,
        fps=0.0,
        latency_ms=0.0,
        detected=False,
    )
