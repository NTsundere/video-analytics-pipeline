from datetime import datetime
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CameraCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    source: str = Field(max_length=500)

    @field_validator("source")
    @classmethod
    def validate_source(cls, value: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme not in {"rtsp", "rtsps"} or not parsed.hostname:
            raise ValueError("source must be an rtsp:// or rtsps:// URL")
        return value


class CameraOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    source: str
    status: str


class EventOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    camera_id: int
    object_class: str
    confidence: float
    bbox: list[float]
    timestamp: datetime
