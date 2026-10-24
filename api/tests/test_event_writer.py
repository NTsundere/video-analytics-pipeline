import asyncio

from app.services import event_writer as module
from app.services.event_writer import EventWriter


class FakeSession:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def make_writer(**kwargs):
    return EventWriter(lambda: FakeSession(), **kwargs)


async def test_events_are_flushed_on_stop(monkeypatch):
    batches = []

    async def fake_log(db, events):
        batches.append(list(events))

    monkeypatch.setattr(module.crud, "log_events_bulk", fake_log)
    writer = make_writer(flush_interval=0.05)
    writer.start()
    writer.submit([{"n": i} for i in range(5)])
    await writer.stop()
    assert sum(len(batch) for batch in batches) == 5


async def test_batches_respect_batch_size(monkeypatch):
    batches = []

    async def fake_log(db, events):
        batches.append(list(events))

    monkeypatch.setattr(module.crud, "log_events_bulk", fake_log)
    writer = make_writer(flush_interval=0.05, batch_size=10)
    writer.start()
    writer.submit([{"n": i} for i in range(25)])
    await writer.stop()
    assert sum(len(batch) for batch in batches) == 25
    assert all(len(batch) <= 10 for batch in batches)


def test_full_queue_drops_and_counts_events():
    writer = make_writer(queue_size=2)
    writer.submit([{"n": i} for i in range(5)])
    assert writer.dropped == 3


async def test_database_error_does_not_stop_the_writer(monkeypatch):
    calls = []

    async def flaky(db, events):
        calls.append(list(events))
        if len(calls) == 1:
            raise RuntimeError("db down")

    monkeypatch.setattr(module.crud, "log_events_bulk", flaky)
    writer = make_writer(flush_interval=0.01)
    writer.start()
    writer.submit([{"n": 1}])
    await asyncio.sleep(0.1)
    writer.submit([{"n": 2}])
    await writer.stop()
    assert len(calls) == 2
