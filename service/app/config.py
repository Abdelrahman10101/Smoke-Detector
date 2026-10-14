"""Central configuration for the smoking detection service."""
from pathlib import Path
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Models & Runtime
    MODELS_DIR: Path = Path("models")
    DETECTOR_MODEL: str = "yolo26n.onnx"
    CLASSIFIER_MODEL: str = "smoking_efficientnet_b0.onnx"
    MODEL_META: str = "model_meta.json"
    MANIFEST_FILE: str = "manifest.json"
    
    REQUIRE_GPU: bool = False
    ORT_THREADS: int = 4
    
    # Video & Processing
    PROCESS_FPS: float = 5.0
    MAX_SESSIONS: int = 2
    VIDEOS_DIR: Path = Path("data/videos")
    UPLOADS_DIR: Path = Path("data/uploads")
    MAX_UPLOAD_MB: int = 500
    EVENTS_DIR: Path = Path("data/events")
    SNAPSHOT_RETENTION_DAYS: int = 7
    
    CCTV_SIM: bool = False
    DASH_TOKEN: str = ""
    
    # Motion Gating
    MOTION_GATE: bool = True
    MOTION_THRESHOLD: float = 1.5  # Mean absolute diff between grey frames
    
    # Detection
    DET_CONF: float = 0.35
    DET_IMGSZ: int = 640
    DET_CLASS_PERSON: int = 0
    
    # Cropping
    CROP_MODE: Literal["person", "upper_body", "full_frame"] = "person"
    CROP_MARGIN: float = 0.10
    MIN_CROP_PX: int = 64
    MAX_PERSONS: int = 10
    
    # Classification & Decision
    SMOKING_THRESHOLD: float = 0.5
    VOTE_K: int = 3
    VOTE_N: int = 5
    CLS_STRIDE: int = 3
    COOLDOWN_S: float = 60.0
    
    # Alerts / Sinks
    EVENT_SINK: str = "dashboard"
    WEBHOOK_URL: str = ""


# Default singleton instance
config = AppConfig()
