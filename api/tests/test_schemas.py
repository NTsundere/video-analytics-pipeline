import pytest
from pydantic import ValidationError

from app.schemas import CameraCreate


@pytest.mark.parametrize("source", ["rtsp://host:554/stream", "rtsps://host/stream"])
def test_accepts_rtsp_sources(source):
    assert CameraCreate(name="cam", source=source).source == source


@pytest.mark.parametrize(
    "source",
    ["http://host/stream", "file:///etc/passwd", "rtsp://", "not a url", ""],
)
def test_rejects_other_sources(source):
    with pytest.raises(ValidationError):
        CameraCreate(name="cam", source=source)


def test_rejects_empty_name():
    with pytest.raises(ValidationError):
        CameraCreate(name="", source="rtsp://host/stream")
