from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app import crud, schemas
from app.database import get_db

router = APIRouter(prefix="/cameras", tags=["cameras"])


@router.post("", response_model=schemas.CameraOut, status_code=status.HTTP_201_CREATED)
async def create_camera(data: schemas.CameraCreate, db: AsyncSession = Depends(get_db)):
    try:
        return await crud.create_camera(db, data)
    except crud.CameraExistsError:
        raise HTTPException(status_code=409, detail="camera with this source already exists")


@router.get("", response_model=list[schemas.CameraOut])
async def list_cameras(db: AsyncSession = Depends(get_db)):
    return await crud.get_cameras(db)


@router.get("/{camera_id}", response_model=schemas.CameraOut)
async def get_camera(camera_id: int, db: AsyncSession = Depends(get_db)):
    camera = await crud.get_camera(db, camera_id)
    if camera is None:
        raise HTTPException(status_code=404, detail="camera not found")
    return camera
