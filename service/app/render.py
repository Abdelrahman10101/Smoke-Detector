"""Visual rendering for video overlays and MJPEG frame encoding."""
from typing import List, Dict, Any, Optional
import cv2
import numpy as np

from .cctv_transforms import cctv_degrade

FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_overlay(
    rgb_frame: np.ndarray,
    detections: List[Dict[str, Any]],
    fps_display: float = 0.0,
    timestamp_s: float = 0.0,
    simulate_cctv: bool = False
) -> np.ndarray:
    """Draw bounding boxes, probability bars, track badges, and alerts on the frame."""
    frame = rgb_frame.copy()
    h, w = frame.shape[:2]

    # Optional CCTV degradation for previewing
    if simulate_cctv:
        rng = np.random.default_rng(int(timestamp_s * 100))
        frame = cctv_degrade(frame, rng)

    has_alert = False

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        track_id = det["track_id"]
        p_smoking = det["p_smoking"]
        is_alert = det.get("is_alert", False)
        is_smoking = det.get("is_smoking", False)

        if is_alert:
            has_alert = True
            color = (255, 45, 85)   # Vibrant Red (RGB)
            status_text = f"TRACK #{track_id} | SMOKING ALERT ({p_smoking*100:.1f}%)"
            box_thick = 3
        elif is_smoking:
            color = (255, 179, 0)   # Amber (RGB)
            status_text = f"TRACK #{track_id} | POSITIVE ({p_smoking*100:.1f}%)"
            box_thick = 2
        else:
            color = (0, 230, 118)   # Emerald Green (RGB)
            status_text = f"TRACK #{track_id} | SAFE ({p_smoking*100:.1f}%)"
            box_thick = 2

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thick, cv2.LINE_AA)

        # Label badge background
        scale = 0.55
        thick = 1
        (tw, th), baseline = cv2.getTextSize(status_text, FONT, scale, thick)
        badge_y1 = max(0, y1 - th - 10)
        badge_y2 = y1
        badge_x2 = min(w, x1 + tw + 12)

        # Draw filled pill background
        cv2.rectangle(frame, (x1, badge_y1), (badge_x2, badge_y2), color, -1)
        # Text in black or white depending on contrast
        text_color = (0, 0, 0) if not is_alert else (255, 255, 255)
        cv2.putText(
            frame, status_text, (x1 + 6, y1 - 6),
            FONT, scale, text_color, thick, cv2.LINE_AA
        )

    # Global Alert Banner at top if any person is in active alert
    if has_alert:
        banner_h = 44
        cv2.rectangle(frame, (0, 0), (w, banner_h), (220, 20, 60), -1)
        banner_msg = " [!] WARNING: SMOKING ACTIVITY DETECTED "
        (btw, bth), _ = cv2.getTextSize(banner_msg, FONT, 0.8, 2)
        cv2.putText(
            frame, banner_msg, ((w - btw) // 2, 30),
            FONT, 0.8, (255, 255, 255), 2, cv2.LINE_AA
        )

    # HUD info bar at the bottom
    hud_h = 30
    cv2.rectangle(frame, (0, h - hud_h), (w, h), (18, 18, 24), -1)
    hud_text = (
        f"Time: {timestamp_s:.1f}s | "
        f"Proc FPS: {fps_display:.1f} | "
        f"Persons: {len(detections)}"
    )
    cv2.putText(
        frame, hud_text, (12, h - 10),
        FONT, 0.45, (200, 200, 200), 1, cv2.LINE_AA
    )

    return frame


def encode_jpeg(rgb_frame: np.ndarray, quality: int = 80) -> bytes:
    """Encode RGB image as JPEG bytes for MJPEG streaming."""
    bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("Failed to encode frame as JPEG")
    return buf.tobytes()
