import asyncio
import logging
from typing import Optional

from app import crud

logger = logging.getLogger(__name__)


class EventWriter:
    def __init__(self, session_factory, queue_size: int = 1000, flush_interval: float = 0.5, batch_size: int = 200):
        self._session_factory = session_factory
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=queue_size)
        self._flush_interval = flush_interval
        self._batch_size = batch_size
        self._task: Optional[asyncio.Task] = None
        self.dropped = 0

    def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name="event-writer")

    def submit(self, events: list[dict]) -> None:
        for index, event in enumerate(events):
            try:
                self._queue.put_nowait(event)
            except asyncio.QueueFull:
                self.dropped += len(events) - index
                logger.warning("event queue is full, dropped events: %s", self.dropped)
                return

    async def stop(self) -> None:
        if self._task is None:
            return
        await self._queue.put(None)
        try:
            await asyncio.wait_for(self._task, timeout=10)
        except asyncio.TimeoutError:
            logger.warning("event writer did not stop in time")

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        stopping = False
        while not stopping:
            first = await self._queue.get()
            if first is None:
                break
            batch = [first]
            deadline = loop.time() + self._flush_interval
            while len(batch) < self._batch_size:
                timeout = deadline - loop.time()
                if timeout <= 0:
                    break
                try:
                    item = await asyncio.wait_for(self._queue.get(), timeout)
                except asyncio.TimeoutError:
                    break
                if item is None:
                    stopping = True
                    break
                batch.append(item)
            await self._flush(batch)

    async def _flush(self, batch: list[dict]) -> None:
        try:
            async with self._session_factory() as db:
                await crud.log_events_bulk(db, batch)
        except Exception:
            logger.exception("failed to write %d events", len(batch))
