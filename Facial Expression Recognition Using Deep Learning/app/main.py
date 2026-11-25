"""FastAPI demo app.

  set FER_CHECKPOINT=outputs/runs/resnet18_finetune/seed42/best.pt     (PowerShell: $env:FER_CHECKPOINT=...)
  uvicorn app.main:app --port 8000

Privacy by design: uploaded images are decoded and processed in memory only - never written to disk,
never logged. An explicit consent flag is required server-side (not just in the UI). The output is a
probability over 7 facial-expression *categories* for the pixels shown; it is NOT an assessment of a
person's emotional state.
"""
from __future__ import annotations

import base64
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from fer.inference import Predictor

STATIC = Path(__file__).parent / "static"
MAX_BYTES = 5 * 1024 * 1024
MAX_SIDE = 4096
DISCLAIMER = ("This tool outputs the probability that the visible facial configuration resembles one of 7 "
              "expression categories learned from FER-2013. It does not measure or infer a person's actual emotion, "
              "can be wrong, and performs unevenly across people and image conditions.")

state: dict = {"predictor": None, "lock": threading.Lock(), "cascade": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    ckpt = os.environ.get("FER_CHECKPOINT")
    if ckpt and Path(ckpt).is_file():
        state["predictor"] = Predictor(ckpt, device=os.environ.get("FER_DEVICE", "cpu"))
    state["cascade"] = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    yield


app = FastAPI(title="Facial Expression Recognition (research demo)", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


def _png_b64(arr: np.ndarray) -> str:
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR) if arr.ndim == 3 else arr
    ok, buf = cv2.imencode(".png", bgr)
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


def crop_face(gray: np.ndarray, use_detector: bool):
    """Largest Haar-detected face (+10% margin), else centered square. Returns (crop, warning|None)."""
    h, w = gray.shape
    if use_detector:
        faces = state["cascade"].detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
        if len(faces):
            x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
            m = int(0.1 * max(fw, fh))
            x0, y0, x1, y1 = max(0, x - m), max(0, y - m), min(w, x + fw + m), min(h, y + fh + m)
            return gray[y0:y1, x0:x1], None
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    warn = "No face detected; using the centered square of the image. Results may be unreliable." if use_detector else None
    return gray[y0:y0 + s, x0:x0 + s], warn


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": state["predictor"] is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...), consent: bool = Form(False), use_face_detection: bool = Form(True)):
    if not consent:
        raise HTTPException(400, "Consent is required to process an image.")
    if state["predictor"] is None:
        raise HTTPException(503, "No model loaded. Set FER_CHECKPOINT to a trained best.pt and restart.")
    raw = await file.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, "Image too large (max 5 MB).")
    gray = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
    del raw
    if gray is None:
        raise HTTPException(400, "Could not decode the file as an image.")
    if max(gray.shape) > MAX_SIDE:
        raise HTTPException(413, f"Image too large (max {MAX_SIDE}px per side).")

    def work():
        face, warn = crop_face(gray, use_face_detection)
        with state["lock"]:
            return face, warn, state["predictor"].predict(face)

    face, warn, res = await run_in_threadpool(work)
    warnings = [warn] if warn else []
    if res["confidence"] < 0.5:
        warnings.append("Low model confidence: the top class has under 50% probability.")
    body = {
        "prediction": res["prediction"], "confidence": res["confidence"], "probs": res["probs"],
        "face_png": _png_b64(cv2.resize(face, (192, 192), interpolation=cv2.INTER_AREA)),
        "cam_png": _png_b64(res["cam_overlay"]), "warnings": warnings, "disclaimer": DISCLAIMER,
    }
    return JSONResponse(body, headers={"Cache-Control": "no-store"})
