import pytest

from app.config import settings
from app.services.video_source import SourceError, list_video_files, resolve_source


@pytest.fixture
def videos(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "videos_dir", str(tmp_path))
    (tmp_path / "a.mp4").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("x")
    return tmp_path


def test_lists_only_video_files(videos):
    assert list_video_files() == ["a.mp4"]


def test_lists_nothing_when_directory_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "videos_dir", str(tmp_path / "missing"))
    assert list_video_files() == []


def test_resolves_file_inside_videos_dir(videos):
    path, is_file = resolve_source("file:a.mp4")
    assert is_file is True
    assert path == str((videos / "a.mp4").resolve())


@pytest.mark.parametrize("source", ["file:../a.mp4", "file:/etc/passwd", "file:missing.mp4", "file:"])
def test_rejects_paths_outside_or_missing(videos, source):
    with pytest.raises(SourceError):
        resolve_source(source)


def test_rtsp_is_passed_through(videos):
    assert resolve_source("rtsp://host/stream") == ("rtsp://host/stream", False)


def test_rejects_unknown_scheme(videos):
    with pytest.raises(SourceError):
        resolve_source("http://host/stream")
