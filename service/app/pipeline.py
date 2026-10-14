"""End-to-end smoking detection pipeline coordinating detector, classifier, and tracker."""
import time
from typing import List, Dict, Any, Optional, Tuple
import cv2
import numpy as np

from .config import AppConfig
from .detector import PersonDetector
from .classifier import SmokingClassifier
from .tracker import PersonTracker, Track
from .preprocess import extract_person_crop


class SmokingPipeline:
    def __init__(
        self,
        detector: PersonDetector,
        classifier: SmokingClassifier,
        cfg: AppConfig
    ):
        self.detector = detector
        self.classifier = classifier
        self.cfg = cfg
        self.tracker = PersonTracker(cfg)
        
        self.prev_gray: Optional[np.ndarray] = None
        self.last_det_results: List[Dict[str, Any]] = []

    def check_motion(self, rgb_frame: np.ndarray) -> bool:
        """Lightweight frame differencing on downscaled grayscale frame (~0.5ms)."""
        small_gray = cv2.cvtColor(
            cv2.resize(rgb_frame, (160, 90), interpolation=cv2.INTER_LINEAR),
            cv2.COLOR_RGB2GRAY
        )
        if self.prev_gray is None:
            self.prev_gray = small_gray
            return True

        diff = cv2.absdiff(self.prev_gray, small_gray)
        mean_diff = float(np.mean(diff))
        self.prev_gray = small_gray
        return mean_diff >= self.cfg.MOTION_THRESHOLD

    def process_frame(
        self,
        rgb_frame: np.ndarray,
        timestamp_s: float = 0.0,
        override_threshold: Optional[float] = None,
        override_crop_mode: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process a single RGB frame through detector, tracker, and classifier."""
        t_start = time.perf_counter()
        thr = override_threshold if override_threshold is not None else self.cfg.SMOKING_THRESHOLD
        crop_mode = override_crop_mode or self.cfg.CROP_MODE

        # 1. Motion Gating
        has_motion = True
        if self.cfg.MOTION_GATE:
            has_motion = self.check_motion(rgb_frame)

        # 2. Person Detection
        det_latency = 0.0
        if has_motion or not self.last_det_results:
            detections, det_latency = self.detector.detect(rgb_frame)
            self.last_det_results = detections
        else:
            detections = self.last_det_results

        # 3. Tracking association
        tracked_items = self.tracker.update(detections, current_time=timestamp_s)

        # 4. Filter items needing classification & extract crops
        crops_to_classify: List[np.ndarray] = []
        tracks_to_classify: List[Track] = []
        debug_crops: List[Dict[str, Any]] = []

        for track, should_cls, det in tracked_items:
            crop = extract_person_crop(
                rgb_frame,
                track.bbox,
                mode=crop_mode,
                margin=self.cfg.CROP_MARGIN
            )
            debug_crops.append({
                "track_id": track.track_id,
                "crop": crop,
                "bbox": track.bbox
            })
            if should_cls:
                crops_to_classify.append(crop)
                tracks_to_classify.append(track)

        # 5. Batched Classification
        cls_latency = 0.0
        if crops_to_classify:
            probs, cls_latency = self.classifier.classify(crops_to_classify)
            for track, prob in zip(tracks_to_classify, probs):
                track.add_prediction(prob)

        # 6. Evaluate alerts and aggregate results
        frame_detections: List[Dict[str, Any]] = []
        new_alerts: List[Dict[str, Any]] = []

        for track, _, det in tracked_items:
            prob = track.last_prob
            alert_fired = track.check_alert(
                threshold=thr,
                k=self.cfg.VOTE_K,
                cooldown_s=self.cfg.COOLDOWN_S,
                current_time=timestamp_s
            )

            is_smoking = (prob >= thr)
            item_data = {
                "track_id": track.track_id,
                "bbox": track.bbox,
                "det_conf": det["conf"],
                "p_smoking": round(prob, 4),
                "is_smoking": is_smoking,
                "is_alert": track.is_alerting,
                "history": list(track.history)
            }
            frame_detections.append(item_data)

            if alert_fired:
                new_alerts.append({
                    "track_id": track.track_id,
                    "timestamp": timestamp_s,
                    "p_smoking": round(prob, 4),
                    "bbox": track.bbox
                })

        total_latency = (time.perf_counter() - t_start) * 1000.0

        return {
            "timestamp": timestamp_s,
            "persons": frame_detections,
            "new_alerts": new_alerts,
            "debug_crops": debug_crops,
            "latencies": {
                "det_ms": round(det_latency, 2),
                "cls_ms": round(cls_latency, 2),
                "total_ms": round(total_latency, 2)
            }
        }
