from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app import crud, schemas
from app.database import get_db

router = APIRouter(prefix="/events", tags=["events"])


@router.get("", response_model=list[schemas.EventOut])
async def list_events(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    camera_id: Optional[int] = None,
    object_class: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    return await crud.get_events(db, limit, offset, camera_id, object_class)
