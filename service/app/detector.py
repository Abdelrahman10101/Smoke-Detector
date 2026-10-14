"""YOLO26n person detector wrapper with end-to-end NMS-free parsing."""
import time
from typing import List, Dict, Any, Tuple
import numpy as np
import onnxruntime as ort

from .config import AppConfig
from .preprocess import prepare_detector_input, unletterbox_box


class PersonDetector:
    def __init__(self, session: ort.InferenceSession, cfg: AppConfig):
        self.session = session
        self.cfg = cfg
        self.input_name = session.get_inputs()[0].name
        self.output_name = session.get_outputs()[0].name
        
        # Verify output shape matches end2end (1, 300, 6)
        out_shape = session.get_outputs()[0].shape
        if len(out_shape) == 3 and out_shape[-1] == 6:
            self.is_end2end = True
        else:
            self.is_end2end = False

    def detect(
        self,
        rgb_img: np.ndarray,
        conf_threshold: float = None
    ) -> Tuple[List[Dict[str, Any]], float]:
        """Detect persons in RGB image.
        
        Returns:
            detections: List of dicts with {"bbox": [x1, y1, x2, y2], "conf": float, "cls": 0}
            latency_ms: Inference time in milliseconds
        """
        conf_thr = conf_threshold if conf_threshold is not None else self.cfg.DET_CONF
        h_orig, w_orig = rgb_img.shape[:2]
        
        # 1. Preprocess: letterbox + scale/pad tracking
        tensor, scale, pad = prepare_detector_input(rgb_img, imgsz=self.cfg.DET_IMGSZ)
        
        # 2. Run inference
        t0 = time.perf_counter()
        outputs = self.session.run([self.output_name], {self.input_name: tensor})
        latency_ms = (time.perf_counter() - t0) * 1000.0
        
        raw_preds = outputs[0]  # shape: (1, 300, 6)
        
        detections: List[Dict[str, Any]] = []
        if self.is_end2end:
            preds = raw_preds[0]  # (300, 6)
            for row in preds:
                conf = float(row[4])
                if conf < conf_thr:
                    continue
                cls_id = int(round(float(row[5])))
                if cls_id != self.cfg.DET_CLASS_PERSON:
                    continue
                
                # Unletterbox coordinates to original image dimensions
                x1, y1, x2, y2 = unletterbox_box(row[:4], scale, pad, (h_orig, w_orig))
                
                # Filter small boxes
                box_w = x2 - x1
                box_h = y2 - y1
                if min(box_w, box_h) < self.cfg.MIN_CROP_PX:
                    continue
                
                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "conf": round(conf, 4),
                    "cls": cls_id
                })
        else:
            # Fallback for non-end2end output (1, 84, 8400)
            # Row format: [cx, cy, w, h, class_probs...]
            preds = raw_preds[0].T  # (8400, 84)
            person_scores = preds[:, 4]  # class 0 = person
            mask = person_scores >= conf_thr
            filtered = preds[mask]
            
            # Simple greedy NMS if needed
            boxes = []
            scores = []
            for row in filtered:
                cx, cy, bw, bh = row[:4]
                x1 = cx - bw / 2
                y1 = cy - bh / 2
                x2 = cx + bw / 2
                y2 = cy + bh / 2
                boxes.append(unletterbox_box([x1, y1, x2, y2], scale, pad, (h_orig, w_orig)))
                scores.append(float(row[4]))
                
            if boxes:
                import cv2
                indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thr, 0.45)
                for idx in indices:
                    i = int(idx)
                    b = boxes[i]
                    if min(b[2] - b[0], b[3] - b[1]) >= self.cfg.MIN_CROP_PX:
                        detections.append({
                            "bbox": b,
                            "conf": round(scores[i], 4),
                            "cls": 0
                        })

        # Sort by confidence descending and cap at MAX_PERSONS
        detections.sort(key=lambda d: d["conf"], reverse=True)
        if len(detections) > self.cfg.MAX_PERSONS:
            detections = detections[:self.cfg.MAX_PERSONS]
            
        return detections, latency_ms
