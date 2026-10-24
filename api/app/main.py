import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app import crud
from app.config import settings
from app.database import AsyncSessionLocal, engine, init_db
from app.routers import cameras, events, metrics, websocket
from app.services.event_writer import EventWriter
from app.services.inference_client import InferenceClient
from app.services.video_source import list_video_files
from app.services.worker_registry import WorkerRegistry

logging.basicConfig(
    level=settings.log_level.upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    async with AsyncSessionLocal() as db:
        await crud.reset_camera_statuses(db)
        await crud.seed_file_cameras(db, list_video_files())

    inference = InferenceClient(settings.inference_url)
    writer = EventWriter(AsyncSessionLocal)
    writer.start()
    registry = WorkerRegistry(inference, writer, AsyncSessionLocal)
    app.state.registry = registry
    app.state.writer = writer
    logger.info("application started")

    try:
        yield
    finally:
        await registry.shutdown()
        await writer.stop()
        await inference.aclose()
        await engine.dispose()
        logger.info("application stopped")


app = FastAPI(title="Video Analytics Pipeline", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "healthy"}


app.include_router(cameras.router, prefix="/api")
app.include_router(events.router, prefix="/api")
app.include_router(metrics.router, prefix="/api")
app.include_router(websocket.router)

frontend_dir = Path(settings.frontend_dir)
if frontend_dir.is_dir():
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
