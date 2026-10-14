"""Video analysis session manager with realtime/fast pacing and MJPEG distribution."""
import asyncio
import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, Any, Optional

import cv2
import numpy as np

from .config import AppConfig
from .sources import FrameSource, VideoFileSource
from .pipeline import SmokingPipeline
from .render import draw_overlay, encode_jpeg
from .events import EventManager

logger = logging.getLogger(__name__)


class AnalysisSession:
    def __init__(
        self,
        session_id: str,
        source: FrameSource,
        pipeline: SmokingPipeline,
        event_manager: EventManager,
        cfg: AppConfig,
        mode: str = "realtime",
        loop: bool = False
    ):
        self.session_id = session_id
        self.source = source
        self.pipeline = pipeline
        self.event_manager = event_manager
        self.cfg = cfg
        self.mode = mode
        self.loop = loop

        self.status = "running"
        self.current_time_s = 0.0
        self.processed_fps = 0.0
        
        # State locks and image buffers
        self._lock = threading.Lock()
        self.latest_frame_jpeg: Optional[bytes] = None
        self.latest_crops_jpeg: Optional[bytes] = None
        self.latest_meta: Dict[str, Any] = {}

        # Runtime tuning overrides
        self.override_threshold: Optional[float] = None
        self.override_crop_mode: Optional[str] = None
        self.override_process_fps: Optional[float] = None
        self.cctv_sim: bool = cfg.CCTV_SIM

        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()  # Not paused initially

        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def _run_loop(self):
        logger.info(f"Session {self.session_id} worker started.")
        source_info = self.source.get_info()
        video_fps = source_info.get("fps", 25.0)
        source_name = source_info.get("name", "video")

        last_inference_time = 0.0
        last_detections = []
        fps_tracker_time = time.perf_counter()
        frames_counted = 0

        while not self._stop_event.is_set():
            self._pause_event.wait()
            if self._stop_event.is_set():
                break

            target_process_fps = self.override_process_fps or self.cfg.PROCESS_FPS
            infer_interval = 1.0 / max(0.5, target_process_fps)

            t_loop_start = time.perf_counter()
            ok, rgb_frame, video_ts = self.source.read_frame()
            if not ok:
                logger.info(f"Session {self.session_id} reached end of stream.")
                self.status = "completed"
                break

            self.current_time_s = video_ts

            # Determine whether to run inference on this frame
            now = time.perf_counter()
            should_infer = (now - last_inference_time) >= infer_interval

            new_alerts = []
            latencies = {"det_ms": 0.0, "cls_ms": 0.0, "total_ms": 0.0}
            debug_crops = []

            if should_infer:
                last_inference_time = now
                res = self.pipeline.process_frame(
                    rgb_frame,
                    timestamp_s=video_ts,
                    override_threshold=self.override_threshold,
                    override_crop_mode=self.override_crop_mode
                )
                last_detections = res["persons"]
                new_alerts = res["new_alerts"]
                latencies = res["latencies"]
                debug_crops = res.get("debug_crops", [])

                # Trigger alert events
                for alert in new_alerts:
                    self.event_manager.record_alert(
                        session_id=self.session_id,
                        video_name=source_name,
                        timestamp_s=video_ts,
                        track_id=alert["track_id"],
                        p_smoking=alert["p_smoking"],
                        bbox=alert["bbox"],
                        rgb_frame=rgb_frame
                    )

            # Render annotated frame
            rendered = draw_overlay(
                rgb_frame,
                last_detections,
                fps_display=self.processed_fps,
                timestamp_s=video_ts,
                simulate_cctv=self.cctv_sim
            )

            # Build debug crops collage if available
            crops_jpeg = None
            if debug_crops:
                collage_parts = []
                for dc in debug_crops[:4]:
                    crop = dc["crop"]
                    # Resize to standard 160x160 thumbnail
                    thumb = cv2.resize(crop, (160, 160), interpolation=cv2.INTER_LINEAR)
                    collage_parts.append(thumb)
                if collage_parts:
                    crops_bar = np.hstack(collage_parts)
                    try:
                        crops_jpeg = encode_jpeg(crops_bar, quality=75)
                    except Exception:
                        pass

            try:
                frame_jpeg = encode_jpeg(rendered, quality=80)
                with self._lock:
                    self.latest_frame_jpeg = frame_jpeg
                    if crops_jpeg:
                        self.latest_crops_jpeg = crops_jpeg
                    self.latest_meta = {
                        "timestamp_s": round(video_ts, 2),
                        "persons": last_detections,
                        "processed_fps": round(self.processed_fps, 1),
                        "latencies": latencies,
                        "status": self.status
                    }
            except Exception as e:
                logger.error(f"Render/encode error in session {self.session_id}: {e}")

            # FPS calculation
            frames_counted += 1
            if time.perf_counter() - fps_tracker_time >= 1.0:
                self.processed_fps = frames_counted / (time.perf_counter() - fps_tracker_time)
                frames_counted = 0
                fps_tracker_time = time.perf_counter()

            # Pacing
            if self.mode == "realtime":
                # Sleep to maintain nominal video FPS playback rate
                elapsed = time.perf_counter() - t_loop_start
                sleep_time = (1.0 / max(1.0, video_fps)) - elapsed
                if sleep_time > 0.001:
                    time.sleep(sleep_time)

        self.source.release()
        logger.info(f"Session {self.session_id} worker stopped.")

    def pause(self):
        self._pause_event.clear()
        self.status = "paused"

    def resume(self):
        self._pause_event.set()
        self.status = "running"

    def stop(self):
        self.status = "stopped"
        self._stop_event.set()
        self._pause_event.set()  # Unblock if paused
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def get_status(self) -> dict:
        with self._lock:
            meta = dict(self.latest_meta)
        return {
            "session_id": self.session_id,
            "status": self.status,
            "mode": self.mode,
            "current_time_s": round(self.current_time_s, 2),
            "processed_fps": round(self.processed_fps, 1),
            "meta": meta
        }


class SessionManager:
    def __init__(self, cfg: AppConfig, pipeline: SmokingPipeline, event_manager: EventManager):
        self.cfg = cfg
        self.pipeline = pipeline
        self.event_manager = event_manager
        self.sessions: Dict[str, AnalysisSession] = {}

    def create_session(
        self,
        video_path: Path,
        mode: str = "realtime",
        loop: bool = True
    ) -> AnalysisSession:
        # Enforce max sessions
        active = [s for s in self.sessions.values() if s.status in ("running", "paused")]
        if len(active) >= self.cfg.MAX_SESSIONS:
            # Stop the oldest active session
            active[0].stop()

        session_id = str(uuid.uuid4())[:8]
        source = VideoFileSource(video_path, loop=loop)
        session = AnalysisSession(
            session_id=session_id,
            source=source,
            pipeline=self.pipeline,
            event_manager=self.event_manager,
            cfg=self.cfg,
            mode=mode,
            loop=loop
        )
        self.sessions[session_id] = session
        return session

    def get(self, session_id: str) -> Optional[AnalysisSession]:
        return self.sessions.get(session_id)

    def list_sessions(self) -> list:
        return [s.get_status() for s in self.sessions.values()]

    def stop_all(self):
        for s in self.sessions.values():
            s.stop()
        self.sessions.clear()
