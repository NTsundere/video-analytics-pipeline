from app.services import stats as stats_module
from app.services.stats import WorkerStats


def test_first_observation_is_taken_as_is():
    stats = WorkerStats(alpha=0.5)
    stats.observe("inference", 0.010)
    assert stats.snapshot()["stages_ms"]["inference"] == 10.0


def test_observations_are_smoothed():
    stats = WorkerStats(alpha=0.5)
    stats.observe("inference", 0.010)
    stats.observe("inference", 0.020)
    assert stats.snapshot()["stages_ms"]["inference"] == 15.0


def test_fps_from_publish_timestamps(monkeypatch):
    clock = {"now": 0.0}
    monkeypatch.setattr(stats_module.time, "monotonic", lambda: clock["now"])
    stats = WorkerStats()
    for step in range(3):
        clock["now"] = step * 0.1
        stats.frame_published()
    assert round(stats.fps(), 1) == 10.0
    assert stats.frames_processed == 3


def test_fps_is_zero_when_stream_stalls(monkeypatch):
    clock = {"now": 0.0}
    monkeypatch.setattr(stats_module.time, "monotonic", lambda: clock["now"])
    stats = WorkerStats()
    stats.frame_published()
    clock["now"] = 0.1
    stats.frame_published()
    clock["now"] = 10.0
    assert stats.fps() == 0.0
