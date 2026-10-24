import asyncio

from app.config import settings
from app.models import Camera
from app.services.camera_worker import CameraWorker
from app.services.event_writer import EventWriter
from app.services.inference_client import InferenceClient
from app.services.video_source import resolve_source


class WorkerLimitError(Exception):
    pass


class WorkerRegistry:
    def __init__(self, inference: InferenceClient, writer: EventWriter, session_factory):
        self._inference = inference
        self._writer = writer
        self._session_factory = session_factory
        self._workers: dict[int, CameraWorker] = {}
        self._lock = asyncio.Lock()

    async def acquire(self, camera: Camera) -> CameraWorker:
        async with self._lock:
            worker = self._workers.get(camera.id)
            if worker is not None and worker.is_alive:
                worker.subscribers += 1
                return worker
            if worker is None and len(self._workers) >= settings.max_workers:
                raise WorkerLimitError(camera.id)
            source, is_file = resolve_source(camera.source)
            worker = CameraWorker(
                camera.id,
                source,
                is_file,
                self._inference,
                self._writer,
                self._session_factory,
            )
            worker.subscribers = 1
            self._workers[camera.id] = worker
            worker.start()
            worker.add_done_callback(self._discard)
            return worker

    async def release(self, worker: CameraWorker) -> None:
        async with self._lock:
            worker.subscribers -= 1
            last = worker.subscribers <= 0
            if last:
                self._discard(worker)
        if last:
            await worker.stop()

    def snapshot(self) -> list[dict]:
        return [worker.snapshot() for worker in self._workers.values()]

    async def shutdown(self) -> None:
        async with self._lock:
            workers = list(self._workers.values())
            self._workers.clear()
        await asyncio.gather(*(worker.stop() for worker in workers))

    def _discard(self, worker: CameraWorker) -> None:
        if self._workers.get(worker.camera_id) is worker:
            del self._workers[worker.camera_id]
