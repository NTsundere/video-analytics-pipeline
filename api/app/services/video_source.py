from pathlib import Path

from app.config import settings

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov"}


class SourceError(Exception):
    pass


def list_video_files() -> list[str]:
    root = Path(settings.videos_dir)
    if not root.is_dir():
        return []
    return sorted(
        path.name
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def resolve_source(source: str) -> tuple[str, bool]:
    if source.startswith(("rtsp://", "rtsps://")):
        return source, False
    if source.startswith("file:"):
        root = Path(settings.videos_dir).resolve()
        path = (root / source[len("file:"):]).resolve()
        if root not in path.parents or not path.is_file():
            raise SourceError(source)
        return str(path), True
    raise SourceError(source)
