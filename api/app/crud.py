from typing import Optional

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Camera, Event
from app.schemas import CameraCreate


class CameraExistsError(Exception):
    pass


async def create_camera(db: AsyncSession, data: CameraCreate) -> Camera:
    camera = Camera(name=data.name, source=data.source)
    db.add(camera)
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise CameraExistsError(data.source)
    await db.refresh(camera)
    return camera


async def get_cameras(db: AsyncSession) -> list[Camera]:
    result = await db.execute(select(Camera).order_by(Camera.id))
    return list(result.scalars().all())


async def get_camera(db: AsyncSession, camera_id: int) -> Optional[Camera]:
    result = await db.execute(select(Camera).where(Camera.id == camera_id))
    return result.scalar_one_or_none()


async def seed_file_cameras(db: AsyncSession, filenames: list[str]) -> None:
    if not filenames:
        return
    rows = [
        {"name": name.rsplit(".", 1)[0][:100], "source": f"file:{name}", "status": "offline"}
        for name in filenames
    ]
    stmt = pg_insert(Camera).values(rows).on_conflict_do_nothing(index_elements=["source"])
    await db.execute(stmt)
    await db.commit()


async def reset_camera_statuses(db: AsyncSession) -> None:
    await db.execute(update(Camera).values(status="offline"))
    await db.commit()


async def set_camera_status(db: AsyncSession, camera_id: int, status: str) -> None:
    await db.execute(update(Camera).where(Camera.id == camera_id).values(status=status))
    await db.commit()


async def get_events(
    db: AsyncSession,
    limit: int = 100,
    offset: int = 0,
    camera_id: Optional[int] = None,
    object_class: Optional[str] = None,
) -> list[Event]:
    stmt = select(Event)
    if camera_id is not None:
        stmt = stmt.where(Event.camera_id == camera_id)
    if object_class is not None:
        stmt = stmt.where(Event.object_class == object_class)
    stmt = stmt.order_by(Event.timestamp.desc(), Event.id.desc()).limit(limit).offset(offset)
    result = await db.execute(stmt)
    return list(result.scalars().all())


async def log_events_bulk(db: AsyncSession, events: list[dict]) -> None:
    db.add_all([Event(**data) for data in events])
    await db.commit()
