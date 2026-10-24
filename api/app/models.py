from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.database import Base


class Camera(Base):
    __tablename__ = "cameras"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    source: Mapped[str] = mapped_column(String(500), unique=True)
    status: Mapped[str] = mapped_column(String(20), default="offline", server_default="offline")


class Event(Base):
    __tablename__ = "events"
    __table_args__ = (Index("ix_events_camera_timestamp", "camera_id", "timestamp"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    camera_id: Mapped[int] = mapped_column(ForeignKey("cameras.id", ondelete="CASCADE"))
    object_class: Mapped[str] = mapped_column(String(50))
    confidence: Mapped[float] = mapped_column(Float)
    bbox: Mapped[list[float]] = mapped_column(JSONB)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        index=True,
    )
