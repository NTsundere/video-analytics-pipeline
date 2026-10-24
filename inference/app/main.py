import asyncio
import base64
import binascii
import logging
import os
from contextlib import asynccontextmanager

import cv2
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("inference")

SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.5"))
CONCURRENCY = int(os.getenv("INFERENCE_CONCURRENCY", "2"))

weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
CATEGORIES = weights.meta["categories"]

logger.info("Loading ONNX model...")
session = ort.InferenceSession("ssdlite320.onnx", providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

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
    h_orig, w_orig = image_bgr.shape[:2]
    
    img_resized = cv2.resize(image_bgr, (320, 320))
    rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    tensor = rgb.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))
    tensor = np.expand_dims(tensor, axis=0)

    boxes, labels, scores = session.run(None, {input_name: tensor})

    results = []
    for box, label, score in zip(boxes, labels, scores):
        if score < SCORE_THRESHOLD:
            continue
        x1, y1, x2, y2 = box.tolist()
        x1 = x1 * w_orig / 320
        y1 = y1 * h_orig / 320
        x2 = x2 * w_orig / 320
        y2 = y2 * h_orig / 320

        results.append({
            "class": CATEGORIES[int(label)],
            "score": round(float(score), 4),
            "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
        })
    return results

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
    logger.info("ONNX model warmed up, concurrency=%s", CONCURRENCY)
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