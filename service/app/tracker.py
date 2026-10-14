"""Lightweight IoU tracker with temporal K-of-N voting and alert cooldown."""
from collections import deque
from typing import List, Dict, Any, Tuple
import numpy as np

from .config import AppConfig


def compute_iou(box1: List[int], box2: List[int]) -> float:
    """Compute Intersection over Union (IoU) of two boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


class Track:
    def __init__(self, track_id: int, bbox: List[int], max_history: int = 5):
        self.track_id = track_id
        self.bbox = bbox
        self.missing_frames = 0
        self.history = deque(maxlen=max_history)
        self.frames_since_cls = 0
        self.last_prob: float = 0.0
        self.is_alerting = False
        self.last_alert_time: float = -9999.0

    def update_box(self, bbox: List[int]):
        self.bbox = bbox
        self.missing_frames = 0
        self.frames_since_cls += 1

    def add_prediction(self, prob: float):
        self.last_prob = prob
        self.history.append(prob)
        self.frames_since_cls = 0

    def check_alert(self, threshold: float, k: int, cooldown_s: float, current_time: float) -> bool:
        """Evaluate K-of-N temporal voting and cooldown."""
        if len(self.history) < k:
            return False

        positive_count = sum(1 for p in self.history if p >= threshold)
        if positive_count >= k:
            if (current_time - self.last_alert_time) >= cooldown_s:
                self.is_alerting = True
                self.last_alert_time = current_time
                return True
        else:
            self.is_alerting = False
        return False


class PersonTracker:
    def __init__(self, cfg: AppConfig, iou_threshold: float = 0.3, max_missing: int = 15):
        self.cfg = cfg
        self.iou_threshold = iou_threshold
        self.max_missing = max_missing
        self.tracks: Dict[int, Track] = {}
        self.next_track_id = 1
        self.global_last_alert_time = -9999.0

    def update(
        self,
        detections: List[Dict[str, Any]],
        current_time: float = 0.0
    ) -> List[Tuple[Track, bool, Dict[str, Any]]]:
        """Associate detections with existing tracks.
        
        Returns:
            results: List of (Track, should_classify, detection_dict)
        """
        # Increment missing count for all tracks initially
        for track in self.tracks.values():
            track.missing_frames += 1

        matched_tracks = set()
        matched_dets = set()
        associations = []

        if self.tracks and detections:
            # Build IoU matrix
            track_ids = list(self.tracks.keys())
            iou_matrix = np.zeros((len(track_ids), len(detections)), dtype=np.float32)
            
            for i, tid in enumerate(track_ids):
                for j, det in enumerate(detections):
                    iou_matrix[i, j] = compute_iou(self.tracks[tid].bbox, det["bbox"])

            # Greedy matching on highest IoU
            while True:
                max_val = np.max(iou_matrix)
                if max_val < self.iou_threshold:
                    break
                i, j = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                tid = track_ids[i]
                
                matched_tracks.add(tid)
                matched_dets.add(j)
                associations.append((self.tracks[tid], detections[j]))
                
                # Zero out row and column
                iou_matrix[i, :] = -1.0
                iou_matrix[:, j] = -1.0

        # Create new tracks for unmatched detections
        for j, det in enumerate(detections):
            if j not in matched_dets:
                new_track = Track(
                    track_id=self.next_track_id,
                    bbox=det["bbox"],
                    max_history=self.cfg.VOTE_N
                )
                self.tracks[self.next_track_id] = new_track
                associations.append((new_track, det))
                self.next_track_id += 1

        # Purge dead tracks
        dead_ids = [
            tid for tid, track in self.tracks.items()
            if track.missing_frames > self.max_missing
        ]
        for tid in dead_ids:
            del self.tracks[tid]

        # Determine which tracks need classification
        results = []
        for track, det in associations:
            track.update_box(det["bbox"])
            # Classify if first time or stride reached
            should_cls = (len(track.history) == 0) or (track.frames_since_cls >= self.cfg.CLS_STRIDE)
            results.append((track, should_cls, det))

        return results
