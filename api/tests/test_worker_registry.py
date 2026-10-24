import pytest

from app.config import settings
from app.services import worker_registry
from app.services.worker_registry import WorkerLimitError, WorkerRegistry


class FakeCamera:
    def __init__(self, camera_id):
        self.id = camera_id
        self.source = "file:x.mp4"


class FakeWorker:
    instances = []

    def __init__(self, camera_id, source, is_file, inference, writer, session_factory):
        self.camera_id = camera_id
        self.subscribers = 0
        self.alive = True
        self.stopped = False
        FakeWorker.instances.append(self)

    @property
    def is_alive(self):
        return self.alive

    def start(self):
        pass

    def add_done_callback(self, callback):
        pass

    def snapshot(self):
        return {"camera_id": self.camera_id}

    async def stop(self):
        self.stopped = True
        self.alive = False


@pytest.fixture
def registry(monkeypatch):
    FakeWorker.instances = []
    monkeypatch.setattr(worker_registry, "CameraWorker", FakeWorker)
    monkeypatch.setattr(worker_registry, "resolve_source", lambda source: ("/tmp/x.mp4", True))
    monkeypatch.setattr(settings, "max_workers", 2)
    return WorkerRegistry(None, None, None)


async def test_second_subscriber_reuses_worker(registry):
    first = await registry.acquire(FakeCamera(1))
    second = await registry.acquire(FakeCamera(1))
    assert first is second
    assert first.subscribers == 2
    assert len(FakeWorker.instances) == 1


async def test_worker_stops_only_after_last_release(registry):
    worker = await registry.acquire(FakeCamera(1))
    await registry.acquire(FakeCamera(1))
    await registry.release(worker)
    assert worker.stopped is False
    await registry.release(worker)
    assert worker.stopped is True
    assert registry.snapshot() == []


async def test_worker_limit_is_enforced(registry):
    await registry.acquire(FakeCamera(1))
    await registry.acquire(FakeCamera(2))
    with pytest.raises(WorkerLimitError):
        await registry.acquire(FakeCamera(3))


async def test_existing_camera_is_allowed_at_the_limit(registry):
    await registry.acquire(FakeCamera(1))
    await registry.acquire(FakeCamera(2))
    again = await registry.acquire(FakeCamera(2))
    assert again.subscribers == 2


async def test_dead_worker_is_replaced(registry):
    first = await registry.acquire(FakeCamera(1))
    first.alive = False
    second = await registry.acquire(FakeCamera(1))
    assert second is not first
    assert second.subscribers == 1


async def test_shutdown_stops_all_workers(registry):
    first = await registry.acquire(FakeCamera(1))
    second = await registry.acquire(FakeCamera(2))
    await registry.shutdown()
    assert first.stopped and second.stopped
    assert registry.snapshot() == []
