import asyncio
import base64
import binascii
import logging
import os
from contextlib import asynccontextmanager

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from torchvision.models.detection import (
    SSDLite320_MobileNet_V3_Large_Weights,
    ssdlite320_mobilenet_v3_large,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("inference")

SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.5"))
CONCURRENCY = int(os.getenv("INFERENCE_CONCURRENCY", "2"))
TORCH_THREADS = int(os.getenv("TORCH_THREADS", "2"))

torch.set_num_threads(TORCH_THREADS)

weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
model = ssdlite320_mobilenet_v3_large(weights=weights).eval()
CATEGORIES = weights.meta["categories"]

semaphore = asyncio.Semaphore(CONCURRENCY)


class Base64Image(BaseModel):
    image: str


def decode_jpeg(data: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="invalid image")
    return image


def decode_base64(text: str) -> np.ndarray:
    try:
        data = base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=400, detail="invalid base64")
    return decode_jpeg(data)


def decode_raw(data: bytes, height: int, width: int) -> np.ndarray:
    if height <= 0 or width <= 0 or len(data) != height * width * 3:
        raise HTTPException(status_code=400, detail="invalid raw frame")
    return np.frombuffer(data, np.uint8).reshape(height, width, 3)


def detect(image_bgr: np.ndarray) -> list[dict]:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0)
    with torch.inference_mode():
        output = model([tensor])[0]
    keep = output["scores"] >= SCORE_THRESHOLD
    boxes = output["boxes"][keep].tolist()
    labels = output["labels"][keep].tolist()
    scores = output["scores"][keep].tolist()
    return [
        {
            "class": CATEGORIES[label],
            "score": round(score, 4),
            "bbox": [round(value, 1) for value in box],
        }
        for box, label, score in zip(boxes, labels, scores)
    ]


def shape_of(image: np.ndarray) -> dict:
    return {"height": int(image.shape[0]), "width": int(image.shape[1])}


def detect_jpeg(data: bytes) -> list[dict]:
    return detect(decode_jpeg(data))


def detect_b64(text: str) -> list[dict]:
    return detect(decode_base64(text))


def detect_raw(data: bytes, height: int, width: int) -> list[dict]:
    return detect(decode_raw(data, height, width))


def echo_jpeg(data: bytes) -> dict:
    return shape_of(decode_jpeg(data))


def echo_b64(text: str) -> dict:
    return shape_of(decode_base64(text))


def echo_raw(data: bytes, height: int, width: int) -> dict:
    return shape_of(decode_raw(data, height, width))


async def run_limited(func, *args):
    async with semaphore:
        return await run_in_threadpool(func, *args)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await run_limited(detect, np.zeros((320, 320, 3), dtype=np.uint8))
    logger.info("model warmed up, concurrency=%s, torch_threads=%s", CONCURRENCY, TORCH_THREADS)
    yield


app = FastAPI(title="Inference Service", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/detect/binary")
async def detect_binary(file: UploadFile = File(...)):
    data = await file.read()
    return await run_limited(detect_jpeg, data)


@app.post("/detect/base64")
async def detect_base64(payload: Base64Image):
    return await run_limited(detect_b64, payload.image)


@app.post("/detect/raw")
async def detect_raw_frame(
    request: Request,
    x_frame_height: int = Header(...),
    x_frame_width: int = Header(...),
):
    data = await request.body()
    return await run_limited(detect_raw, data, x_frame_height, x_frame_width)


@app.post("/echo/binary")
async def echo_binary(file: UploadFile = File(...)):
    data = await file.read()
    return await run_in_threadpool(echo_jpeg, data)


@app.post("/echo/base64")
async def echo_base64(payload: Base64Image):
    return await run_in_threadpool(echo_b64, payload.image)


@app.post("/echo/raw")
async def echo_raw_frame(
    request: Request,
    x_frame_height: int = Header(...),
    x_frame_width: int = Header(...),
):
    data = await request.body()
    return await run_in_threadpool(echo_raw, data, x_frame_height, x_frame_width)
