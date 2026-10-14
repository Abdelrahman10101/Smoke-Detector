"""Alert logging, snapshot storage, and Server-Sent Events (SSE) broadcaster."""
import asyncio
import json
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Set
import cv2
import numpy as np

from .config import AppConfig


class EventManager:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.events_dir = cfg.EVENTS_DIR
        self.snapshots_dir = self.events_dir / "snapshots"
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        
        self.db_path = self.events_dir / "events.db"
        self._init_db()
        self.subscribers: Set[asyncio.Queue] = set()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    video_name TEXT,
                    timestamp_s REAL,
                    track_id INTEGER,
                    p_smoking REAL,
                    bbox TEXT,
                    snapshot_file TEXT,
                    created_at TEXT
                )
            """)
            conn.commit()

    def record_alert(
        self,
        session_id: str,
        video_name: str,
        timestamp_s: float,
        track_id: int,
        p_smoking: float,
        bbox: List[int],
        rgb_frame: np.ndarray
    ) -> Dict[str, Any]:
        """Record alert to database, save snapshot, and notify SSE subscribers."""
        snapshot_filename = f"alert_{int(time.time()*1000)}_track{track_id}.jpg"
        snapshot_path = self.snapshots_dir / snapshot_filename
        
        # Save snapshot image
        bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(snapshot_path), bgr)

        now_str = datetime.utcnow().isoformat()
        bbox_str = json.dumps(bbox)

        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO alerts (session_id, video_name, timestamp_s, track_id, p_smoking, bbox, snapshot_file, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (session_id, video_name, timestamp_s, track_id, p_smoking, bbox_str, snapshot_filename, now_str))
            alert_id = cur.lastrowid
            conn.commit()

        event_payload = {
            "id": alert_id,
            "session_id": session_id,
            "video_name": video_name,
            "timestamp_s": round(timestamp_s, 2),
            "track_id": track_id,
            "p_smoking": round(p_smoking, 4),
            "bbox": bbox,
            "snapshot_file": snapshot_filename,
            "created_at": now_str
        }

        # Broadcast to any active SSE subscribers
        self.broadcast_sync(event_payload)
        return event_payload

    def list_alerts(self, limit: int = 100) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT * FROM alerts ORDER BY id DESC LIMIT ?
            """, (limit,)).fetchall()
            
            alerts = []
            for r in rows:
                d = dict(r)
                d["bbox"] = json.loads(d["bbox"])
                alerts.append(d)
            return alerts

    def purge_old_snapshots(self):
        """Delete snapshots older than configured retention."""
        cutoff = datetime.utcnow() - timedelta(days=self.cfg.SNAPSHOT_RETENTION_DAYS)
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT id, snapshot_file, created_at FROM alerts WHERE created_at < ?",
                (cutoff.isoformat(),)
            ).fetchall()
            
            for row in rows:
                snap_p = self.snapshots_dir / row[1]
                if snap_p.exists():
                    snap_p.unlink(missing_ok=True)
            
            conn.execute("DELETE FROM alerts WHERE created_at < ?", (cutoff.isoformat(),))
            conn.commit()

    # ---- SSE Subscription ----
    async def subscribe(self) -> asyncio.Queue:
        q = asyncio.Queue()
        self.subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue):
        self.subscribers.discard(q)

    def broadcast_sync(self, data: dict):
        for q in list(self.subscribers):
            try:
                q.put_nowait(data)
            except Exception:
                pass
