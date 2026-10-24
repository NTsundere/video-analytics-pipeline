import logging
import threading
import time
from typing import Callable, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class FrameGrabber(threading.Thread):
    def __init__(self, source: str, is_file: bool, on_frame: Callable[[], None]):
        super().__init__(name="frame-grabber", daemon=True)
        self._source = source
        self._is_file = is_file
        self._on_frame = on_frame
        self._stop_flag = threading.Event()
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._seq = 0

    def latest(self) -> tuple[int, Optional[np.ndarray]]:
        with self._lock:
            return self._seq, self._frame

    def stop(self) -> None:
        self._stop_flag.set()

    def run(self) -> None:
        backoff = 1.0
        while not self._stop_flag.is_set():
            reads = 0
            cap = cv2.VideoCapture(self._source)
            try:
                if cap.isOpened():
                    reads = self._read_loop(cap)
                else:
                    logger.warning("cannot open source %s", self._source)
            finally:
                cap.release()
            if reads > 0:
                backoff = 1.0
                if self._is_file:
                    continue
            if self._stop_flag.wait(backoff):
                break
            if reads == 0:
                backoff = min(backoff * 2, 10.0)

    def _read_loop(self, cap: cv2.VideoCapture) -> int:
        interval = 0.0
        if self._is_file:
            fps = cap.get(cv2.CAP_PROP_FPS)
            interval = 1.0 / fps if 0 < fps <= 120 else 1.0 / 25.0
        reads = 0
        next_at = time.monotonic()
        while not self._stop_flag.is_set():
            ok, frame = cap.read()
            if not ok:
                break
            reads += 1
            if not self._publish(frame):
                self._stop_flag.set()
                break
            if interval:
                next_at += interval
                delay = next_at - time.monotonic()
                if delay > 0:
                    if self._stop_flag.wait(delay):
                        break
                else:
                    next_at = time.monotonic()
        return reads

    def _publish(self, frame: np.ndarray) -> bool:
        with self._lock:
            self._frame = frame
            self._seq += 1
        try:
            self._on_frame()
        except RuntimeError:
            return False
        return True
