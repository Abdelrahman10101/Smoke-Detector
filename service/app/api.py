"""FastAPI application providing dashboard, REST API, MJPEG streams, and SSE alerts."""
import asyncio
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Optional

# Explicit anyio.to_thread import for starlette compatibility
try:
    import anyio.to_thread
except ImportError:
    pass

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import AppConfig, config
from .detector import PersonDetector
from .classifier import SmokingClassifier
from .events import EventManager
from .onnx_session import create_session, verify_manifest, warmup_session
from .pipeline import SmokingPipeline
from .sessions import SessionManager

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Smoking Detection Service",
    description="Edge CCTV & Video Smoking Detection using YOLO26n and EfficientNet-B0 ONNX",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global singletons
def _resolve_models_dir(cfg_dir: Path) -> Path:
    if cfg_dir.is_absolute() and cfg_dir.exists():
        return cfg_dir
    candidates = [
        Path.cwd() / cfg_dir,
        Path(__file__).resolve().parent.parent / cfg_dir,         # Container: /app/models
        Path(__file__).resolve().parent.parent.parent / cfg_dir,  # Local dev: repo/models
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    return (Path(__file__).resolve().parent.parent / cfg_dir).resolve()

models_dir = _resolve_models_dir(config.MODELS_DIR)
event_manager: Optional[EventManager] = None
session_manager: Optional[SessionManager] = None
pipeline: Optional[SmokingPipeline] = None
detector: Optional[PersonDetector] = None
classifier: Optional[SmokingClassifier] = None
is_ready = False
active_providers = []


@app.on_event("startup")
async def startup_event():
    global event_manager, session_manager, pipeline, detector, classifier, is_ready, active_providers
    logger.info("Initializing models and sessions...")
    
    # Ensure data directories exist
    config.VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    config.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    config.EVENTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Manifest verification
    verify_manifest(models_dir)

    # 2. Load ONNX sessions
    det_path = models_dir / config.DETECTOR_MODEL
    cls_path = models_dir / config.CLASSIFIER_MODEL
    
    det_sess = create_session(det_path, config)
    cls_sess = create_session(cls_path, config)
    
    # Warm up sessions
    warmup_session(det_sess)
    warmup_session(cls_sess)
    
    active_providers = det_sess.get_providers()

    # 3. Create components
    detector = PersonDetector(det_sess, config)
    classifier = SmokingClassifier(cls_sess, config)
    pipeline = SmokingPipeline(detector, classifier, config)
    event_manager = EventManager(config)
    session_manager = SessionManager(config, pipeline, event_manager)

    is_ready = True
    logger.info("Service initialization complete and ready.")


@app.on_event("shutdown")
async def shutdown_event():
    global session_manager
    if session_manager:
        session_manager.stop_all()


# ---- Health & Observability ----
@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/readyz")
async def readyz():
    if not is_ready:
        raise HTTPException(status_code=503, detail="Models still loading")
    return {
        "ready": True,
        "detector": config.DETECTOR_MODEL,
        "classifier": config.CLASSIFIER_MODEL,
        "providers": active_providers
    }


@app.get("/version")
async def version():
    manifest_path = models_dir / config.MANIFEST_FILE
    manifest = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except Exception:
            pass
            
    return {
        "service": "Smoking Detection Service",
        "version": "1.0.0",
        "models": manifest.get("models", {}),
        "threshold": config.SMOKING_THRESHOLD,
        "crop_mode": config.CROP_MODE
    }


# ---- Video File Management ----
@app.get("/api/videos")
async def list_videos():
    videos = []
    # Scan both configured videos dir and uploads dir
    for folder in [config.VIDEOS_DIR, config.UPLOADS_DIR]:
        if not folder.exists():
            continue
        for ext in [".mp4", ".avi", ".mov", ".mkv", ".webm"]:
            for f in folder.glob(f"*{ext}"):
                cap = cv2.VideoCapture(str(f))
                fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                dur = round(frames / fps, 1) if fps > 0 else 0.0
                cap.release()
                videos.append({
                    "id": f.name,
                    "filename": f.name,
                    "path": str(f.resolve()),
                    "size_mb": round(f.stat().st_size / (1024 * 1024), 2),
                    "duration_s": dur,
                    "is_upload": (folder == config.UPLOADS_DIR)
                })
    return {"videos": videos}


@app.post("/api/videos")
async def upload_video(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    allowed_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    if ext not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{ext}'. Allowed: {', '.join(allowed_exts)}"
        )

    # Save to uploads dir
    dest = config.UPLOADS_DIR / file.filename
    with dest.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Verify video is readable
    cap = cv2.VideoCapture(str(dest))
    if not cap.isOpened():
        cap.release()
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Corrupt or unreadable video file")
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    dur = round(frames / fps, 1) if fps > 0 else 0.0
    cap.release()

    return {
        "status": "uploaded",
        "video_id": dest.name,
        "filename": dest.name,
        "size_mb": round(dest.stat().st_size / (1024 * 1024), 2),
        "duration_s": dur
    }


# ---- Session Management & Streaming ----
class SessionCreateRequest(BaseModel):
    video_id: str
    mode: str = "realtime"  # "realtime" | "fast"
    loop: bool = True


@app.post("/api/sessions")
async def create_session_endpoint(req: SessionCreateRequest):
    # Locate video file
    video_path = None
    for folder in [config.UPLOADS_DIR, config.VIDEOS_DIR]:
        candidate = folder / req.video_id
        if candidate.exists():
            video_path = candidate
            break
            
    if not video_path:
        raise HTTPException(status_code=404, detail=f"Video '{req.video_id}' not found")

    session = session_manager.create_session(video_path, mode=req.mode, loop=req.loop)
    return session.get_status()


@app.get("/api/sessions")
async def get_sessions():
    return {"sessions": session_manager.list_sessions()}


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    sess = session_manager.get(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")
    return sess.get_status()


@app.post("/api/sessions/{session_id}/pause")
async def pause_session(session_id: str):
    sess = session_manager.get(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")
    sess.pause()
    return {"status": "paused"}


@app.post("/api/sessions/{session_id}/resume")
async def resume_session(session_id: str):
    sess = session_manager.get(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")
    sess.resume()
    return {"status": "running"}


@app.post("/api/sessions/{session_id}/stop")
async def stop_session(session_id: str):
    sess = session_manager.get(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")
    sess.stop()
    return {"status": "stopped"}


@app.get("/api/sessions/{session_id}/stream")
async def mjpeg_stream(session_id: str):
    sess = session_manager.get(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")

    async def frame_generator():
        while sess.status in ("running", "paused"):
            with sess._lock:
                frame_bytes = sess.latest_frame_jpeg

            if frame_bytes:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                )
            await asyncio.sleep(0.04)  # ~25 FPS client refresh

    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/api/sessions/{session_id}/crops")
async def crops_stream(session_id: str):
    sess = session_manager.get(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")

    async def crops_generator():
        while sess.status in ("running", "paused"):
            with sess._lock:
                crop_bytes = sess.latest_crops_jpeg

            if crop_bytes:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + crop_bytes + b"\r\n"
                )
            await asyncio.sleep(0.1)

    return StreamingResponse(
        crops_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


# ---- Live Tuning Configuration ----
class ConfigPatchRequest(BaseModel):
    threshold: Optional[float] = None
    crop_mode: Optional[str] = None
    process_fps: Optional[float] = None
    cctv_sim: Optional[bool] = None
    vote_k: Optional[int] = None
    vote_n: Optional[int] = None
    cooldown_s: Optional[float] = None


@app.get("/api/config")
async def get_config():
    return {
        "threshold": config.SMOKING_THRESHOLD,
        "crop_mode": config.CROP_MODE,
        "process_fps": config.PROCESS_FPS,
        "cctv_sim": config.CCTV_SIM,
        "vote_k": config.VOTE_K,
        "vote_n": config.VOTE_N,
        "cooldown_s": config.COOLDOWN_S
    }


@app.patch("/api/config")
async def patch_config(patch: ConfigPatchRequest):
    if patch.threshold is not None:
        config.SMOKING_THRESHOLD = patch.threshold
    if patch.crop_mode is not None:
        config.CROP_MODE = patch.crop_mode
    if patch.process_fps is not None:
        config.PROCESS_FPS = patch.process_fps
    if patch.cctv_sim is not None:
        config.CCTV_SIM = patch.cctv_sim
    if patch.vote_k is not None:
        config.VOTE_K = patch.vote_k
    if patch.vote_n is not None:
        config.VOTE_N = patch.vote_n
    if patch.cooldown_s is not None:
        config.COOLDOWN_S = patch.cooldown_s

    # Propagate to any active sessions
    for sess in session_manager.sessions.values():
        if patch.threshold is not None:
            sess.override_threshold = patch.threshold
        if patch.crop_mode is not None:
            sess.override_crop_mode = patch.crop_mode
        if patch.process_fps is not None:
            sess.override_process_fps = patch.process_fps
        if patch.cctv_sim is not None:
            sess.cctv_sim = patch.cctv_sim

    return await get_config()


# ---- Events & Snapshots ----
@app.get("/api/events")
async def get_events(limit: int = 100):
    return {"alerts": event_manager.list_alerts(limit=limit)}


@app.get("/api/events/stream")
async def events_sse_stream(request: Request):
    q = await event_manager.subscribe()

    async def event_generator():
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    alert = await asyncio.wait_for(q.get(), timeout=1.0)
                    yield f"data: {json.dumps(alert)}\n\n"
                except asyncio.TimeoutError:
                    # Keep-alive heartbeat
                    yield ": ping\n\n"
        finally:
            event_manager.unsubscribe(q)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/snapshots/{filename}")
async def get_snapshot(filename: str):
    snap_p = event_manager.snapshots_dir / filename
    if not snap_p.exists():
        raise HTTPException(status_code=404, detail="Snapshot not found")
    return Response(content=snap_p.read_bytes(), media_type="image/jpeg")


# ---- Single Image Predict Endpoint ----
@app.post("/v1/predict")
async def predict_single_image(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    res = pipeline.process_frame(rgb, timestamp_s=0.0)
    return {
        "persons": res["persons"],
        "latencies": res["latencies"]
    }


# ---- Serve Dashboard SPA ----
dashboard_dir = Path(__file__).parent.parent / "dashboard"
if dashboard_dir.exists():
    app.mount("/static", StaticFiles(directory=str(dashboard_dir)), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def serve_dashboard():
        index_file = dashboard_dir / "index.html"
        if index_file.exists():
            return HTMLResponse(content=index_file.read_text(encoding="utf-8"))
        return HTMLResponse("<h3>Smoking Detection Service Running</h3>")

