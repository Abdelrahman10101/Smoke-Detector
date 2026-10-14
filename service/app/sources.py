"""Frame sources interface: VideoFileSource for demos, RtspSource for CCTV."""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generator, Optional, Tuple
import cv2
import numpy as np


class FrameSource(ABC):
    @abstractmethod
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray], float]:
        """Read the next frame.
        
        Returns:
            (success, rgb_frame, video_timestamp_seconds)
        """
        pass

    @abstractmethod
    def release(self):
        pass

    @abstractmethod
    def get_info(self) -> dict:
        pass


class VideoFileSource(FrameSource):
    def __init__(self, video_path: Path, loop: bool = False):
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")
        
        self.loop = loop
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {self.video_path}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self.current_frame = 0

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray], float]:
        ok, bgr = self.cap.read()
        if not ok:
            if self.loop and self.total_frames > 0:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.current_frame = 0
                ok, bgr = self.cap.read()
                if not ok:
                    return False, None, 0.0
            else:
                return False, None, 0.0

        self.current_frame += 1
        timestamp = self.current_frame / self.fps
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return True, rgb, timestamp

    def seek(self, frame_num: int):
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            self.current_frame = frame_num

    def release(self):
        if self.cap.isOpened():
            self.cap.release()

    def get_info(self) -> dict:
        duration_s = (self.total_frames / self.fps) if self.fps > 0 else 0.0
        return {
            "name": self.video_path.name,
            "fps": round(self.fps, 2),
            "total_frames": self.total_frames,
            "duration_s": round(duration_s, 2),
            "width": self.width,
            "height": self.height
        }


class RtspSource(FrameSource):
    """Stub for RTSP camera stream support (latest-frame-wins queue, auto-reconnect)."""
    def __init__(self, rtsp_url: str):
        self.rtsp_url = rtsp_url
        self.cap = None

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray], float]:
        # Drop-in implementation when cameras are available
        return False, None, 0.0

    def release(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()

    def get_info(self) -> dict:
        return {"rtsp_url": self.rtsp_url, "type": "rtsp"}
