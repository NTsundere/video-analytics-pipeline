import asyncio
import logging
import time
from typing import Callable, Optional

import cv2
import numpy as np

from app import crud
from app.config import settings
from app.services.event_writer import EventWriter
from app.services.frame_grabber import FrameGrabber
from app.services.inference_client import InferenceClient
from app.services.stats import WorkerStats
from app.services.tracker import IouTracker

logger = logging.getLogger(__name__)
BOX_COLOR = (0, 200, 80)


class CameraWorker:
    def __init__(self, camera_id: int, source: str, is_file: bool, inference: InferenceClient, writer: EventWriter, session_factory):
        self.camera_id = camera_id
        self.subscribers = 0
        self.stats = WorkerStats()
        self._source = source
        self._is_file = is_file
        self._inference = inference
        self._writer = writer
        self._session_factory = session_factory
        self._grabber: Optional[FrameGrabber] = None
        self._stop_event = asyncio.Event()
        self._new_frame = asyncio.Event()
        self._frame_cond = asyncio.Condition()
        self._frame: Optional[bytes] = None
        self._frame_seq = 0
        self._closed = False
        self._marked_online = False
        self._inference_retry_at = 0.0
        self._tracker = IouTracker(settings.track_iou_threshold, settings.track_max_missed, settings.track_min_hits)
        
        self._inference_queue: asyncio.Queue = asyncio.Queue(maxsize=1)
        self._render_queue: asyncio.Queue = asyncio.Queue(maxsize=1)
        
        self._tasks: list[asyncio.Task] = []

    @property
    def is_alive(self) -> bool:
        return not self._closed and len(self._tasks) > 0 and not all(t.done() for t in self._tasks)

    def snapshot(self) -> dict:
        data = self.stats.snapshot()
        data["camera_id"] = self.camera_id
        data["subscribers"] = self.subscribers
        data["alive"] = self.is_alive
        return data

    def start(self) -> None:
        loop = asyncio.get_running_loop()
        self._grabber = FrameGrabber(self._source, self._is_file, lambda: loop.call_soon_threadsafe(self._new_frame.set))
        self._grabber.start()
        
        self._tasks = [
            asyncio.create_task(self._inference_loop(), name=f"inf-{self.camera_id}"),
            asyncio.create_task(self._render_loop(), name=f"ren-{self.camera_id}")
        ]

    def add_done_callback(self, callback: Callable[["CameraWorker"], None]) -> None:
        for task in self._tasks:
            task.add_done_callback(lambda _: callback(self))

    async def stop(self) -> None:
        self._stop_event.set()
        self._new_frame.set()
        if self._grabber is not None:
            self._grabber.stop()
            
        for task in self._tasks:
            task.cancel()
        
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
            
        if self._grabber is not None:
            await asyncio.to_thread(self._grabber.join, 3.0)
            
        await self._close_stream()
        await self._set_status("offline")

    async def next_frame(self, last_seq: int) -> tuple[int, Optional[bytes]]:
        async with self._frame_cond:
            await self._frame_cond.wait_for(lambda: self._frame_seq != last_seq or self._closed)
            if self._closed:
                return last_seq, None
            return self._frame_seq, self._frame

    def _put_latest(self, queue: asyncio.Queue, item: any):
        """Кладет в очередь, вытесняя старый элемент, если очередь полна."""
        try:
            queue.put_nowait(item)
        except asyncio.QueueFull:
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            queue.put_nowait(item)

    async def _inference_loop(self) -> None:
        last_seq = 0
        while not self._stop_event.is_set():
            try:
                await asyncio.wait_for(self._new_frame.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            self._new_frame.clear()
            
            seq, frame = self._grabber.latest()
            if frame is None or seq == last_seq:
                continue
            if last_seq:
                self.stats.frames_dropped += max(0, seq - last_seq - 1)
            last_seq = seq
            
            started = time.perf_counter()
            detections = None
            scale_x = scale_y = 1.0
            
            if time.monotonic() >= self._inference_retry_at:
                try:
                    payload, scale_x, scale_y = await asyncio.to_thread(self._prepare, frame)
                    prepared = time.perf_counter()
                    detections = await self._inference.detect(payload)
                    inferred = time.perf_counter()
                    self.stats.observe("prepare", prepared - started)
                    self.stats.observe("inference", inferred - prepared)
                except Exception as exc:
                    self._inference_retry_at = time.monotonic() + 2.0
                    self.stats.inference_errors += 1
                    logger.warning("inference failed for camera %s: %r", self.camera_id, exc)
                    continue
            else:
                continue

            self._put_latest(self._render_queue, (frame, detections, scale_x, scale_y, started))

    async def _render_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                item = await asyncio.wait_for(self._render_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
                
            frame, detections, scale_x, scale_y, started = item
            inferred = time.perf_counter()
            
            try:
                jpeg, events = await asyncio.to_thread(self._render, frame, detections, scale_x, scale_y)
                rendered = time.perf_counter()
                self.stats.observe("render", rendered - inferred)
                self.stats.observe("total", rendered - started)
                
                if events:
                    self._writer.submit(events)
                    
                await self._publish(jpeg)
                self.stats.frame_published()
                
                if not self._marked_online:
                    self._marked_online = True
                    await self._set_status("online")
            except Exception as exc:
                logger.error("render failed for camera %s: %r", self.camera_id, exc)

    def _prepare(self, frame: np.ndarray) -> tuple[bytes, float, float]:
        height, width = frame.shape[:2]
        if width > settings.target_width:
            new_width = settings.target_width
            new_height = max(1, round(height * new_width / width))
            small = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        else:
            small = frame
        ok, buffer = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, settings.jpeg_quality_inference])
        if not ok:
            raise RuntimeError("jpeg encoding failed")
        return buffer.tobytes(), width / small.shape[1], height / small.shape[0]

    def _render(self, frame: np.ndarray, detections: Optional[list[dict]], scale_x: float, scale_y: float) -> tuple[bytes, list[dict]]:
        annotated = frame.copy()
        events: list[dict] = []
        if detections is not None:
            scaled = []
            for det in detections:
                if det["score"] < settings.score_threshold:
                    continue
                x1, y1, x2, y2 = det["bbox"]
                scaled.append({"class": det["class"], "score": det["score"], "bbox": [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]})
            
            tracks = self._tracker.update(scaled)
            for det, track in zip(scaled, tracks):
                x1, y1, x2, y2 = (int(value) for value in det["bbox"])
                cv2.rectangle(annotated, (x1, y1), (x2, y2), BOX_COLOR, 2)
                cv2.putText(annotated, f"#{track.track_id} {det['class']} {det['score']:.2f}", (x1, max(y1 - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, BOX_COLOR, 2)
            
            events = [{"camera_id": self.camera_id, "object_class": track.object_class, "confidence": track.confidence, "bbox": [round(v, 1) for v in track.bbox]} for track in self._tracker.pop_confirmed()]

        ok, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, settings.jpeg_quality_stream])
        if not ok:
            raise RuntimeError("jpeg encoding failed")
        return buffer.tobytes(), events

    async def _publish(self, jpeg: bytes) -> None:
        async with self._frame_cond:
            self._frame = jpeg
            self._frame_seq += 1
            self._frame_cond.notify_all()

    async def _close_stream(self) -> None:
        async with self._frame_cond:
            self._closed = True
            self._frame_cond.notify_all()

    async def _set_status(self, status: str) -> None:
        try:
            async with self._session_factory() as db:
                await crud.set_camera_status(db, self.camera_id, status)
        except Exception:
            logger.exception("failed to set status %s for camera %s", status, self.camera_id)