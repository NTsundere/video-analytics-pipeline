import time
from collections import deque


class WorkerStats:
    def __init__(self, alpha: float = 0.1):
        self._alpha = alpha
        self._published: deque[float] = deque(maxlen=60)
        self._stage_ms: dict[str, float] = {}
        self.frames_processed = 0
        self.frames_dropped = 0
        self.inference_errors = 0

    def observe(self, stage: str, seconds: float) -> None:
        value = seconds * 1000.0
        previous = self._stage_ms.get(stage)
        self._stage_ms[stage] = value if previous is None else previous + self._alpha * (value - previous)

    def frame_published(self) -> None:
        self.frames_processed += 1
        self._published.append(time.monotonic())

    def fps(self) -> float:
        if len(self._published) < 2:
            return 0.0
        if time.monotonic() - self._published[-1] > 2.0:
            return 0.0
        span = self._published[-1] - self._published[0]
        return (len(self._published) - 1) / span if span > 0 else 0.0

    def snapshot(self) -> dict:
        return {
            "fps": round(self.fps(), 1),
            "frames_processed": self.frames_processed,
            "frames_dropped": self.frames_dropped,
            "inference_errors": self.inference_errors,
            "stages_ms": {name: round(value, 1) for name, value in self._stage_ms.items()},
        }
