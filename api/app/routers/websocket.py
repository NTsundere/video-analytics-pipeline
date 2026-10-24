import asyncio
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app import crud
from app.config import settings
from app.database import AsyncSessionLocal
from app.services.video_source import SourceError
from app.services.worker_registry import WorkerLimitError

logger = logging.getLogger(__name__)

router = APIRouter()


async def wait_disconnect(websocket: WebSocket) -> None:
    while True:
        message = await websocket.receive()
        if message["type"] == "websocket.disconnect":
            return


@router.websocket("/ws/stream/{camera_id}")
async def stream(websocket: WebSocket, camera_id: int):
    registry = websocket.app.state.registry

    async with AsyncSessionLocal() as db:
        camera = await crud.get_camera(db, camera_id)
    if camera is None:
        await websocket.close(code=1008)
        return

    await websocket.accept()

    try:
        worker = await registry.acquire(camera)
    except WorkerLimitError:
        await websocket.close(code=1013)
        return
    except SourceError:
        logger.warning("camera %s has an unusable source", camera_id)
        await websocket.close(code=1008)
        return

    disconnect = asyncio.create_task(wait_disconnect(websocket))
    last_seq = 0
    try:
        while True:
            frame_task = asyncio.create_task(worker.next_frame(last_seq))
            done, _ = await asyncio.wait(
                {frame_task, disconnect},
                timeout=settings.stream_timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if disconnect in done:
                frame_task.cancel()
                break
            if frame_task not in done:
                frame_task.cancel()
                await websocket.close(code=1011)
                break
            last_seq, frame = frame_task.result()
            if frame is None:
                await websocket.close(code=1011)
                break
            await websocket.send_bytes(frame)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        disconnect.cancel()
        await registry.release(worker)
