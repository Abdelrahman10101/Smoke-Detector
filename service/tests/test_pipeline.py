"""Unit and integration tests for the core smoking detection pipeline."""
import sys
from pathlib import Path

# Add service/app to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from app.config import AppConfig
from app.onnx_session import create_session, warmup_session
from app.preprocess import letterbox, unletterbox_box, extract_person_crop
from app.detector import PersonDetector
from app.classifier import SmokingClassifier
from app.tracker import PersonTracker, compute_iou
from app.pipeline import SmokingPipeline


def test_letterbox_unletterbox():
    orig_shape = (720, 1280)
    img = np.zeros((*orig_shape, 3), dtype=np.uint8)
    lb_img, scale, pad = letterbox(img, target_size=640)
    assert lb_img.shape == (640, 640, 3)

    # Test coordinate round-trip
    orig_box = [100, 150, 400, 600]
    # Map to letterbox coordinates manually
    lb_box = [
        orig_box[0] * scale + pad[0],
        orig_box[1] * scale + pad[1],
        orig_box[2] * scale + pad[0],
        orig_box[3] * scale + pad[1]
    ]
    unmapped = unletterbox_box(lb_box, scale, pad, orig_shape)
    assert abs(unmapped[0] - orig_box[0]) <= 2
    assert abs(unmapped[1] - orig_box[1]) <= 2
    assert abs(unmapped[2] - orig_box[2]) <= 2
    assert abs(unmapped[3] - orig_box[3]) <= 2


def test_iou_computation():
    box1 = [10, 10, 50, 50]
    box2 = [10, 10, 50, 50]
    assert compute_iou(box1, box2) == 1.0

    box3 = [60, 60, 100, 100]
    assert compute_iou(box1, box3) == 0.0

    box4 = [30, 10, 70, 50]  # Overlap is 20x40 = 800. Area1=1600, Area2=1600, Union=2400. IoU = 1/3
    assert abs(compute_iou(box1, box4) - (1.0 / 3.0)) < 1e-4


def test_tracker_temporal_voting():
    cfg = AppConfig()
    cfg.VOTE_K = 3
    cfg.VOTE_N = 5
    cfg.COOLDOWN_S = 10.0
    cfg.SMOKING_THRESHOLD = 0.5

    tracker = PersonTracker(cfg)
    det = [{"bbox": [50, 50, 200, 400], "conf": 0.9, "cls": 0}]

    # Step 1: initial frame -> new track created
    results = tracker.update(det, current_time=0.0)
    assert len(results) == 1
    track, should_cls, _ = results[0]
    assert should_cls is True
    track.add_prediction(0.8)  # Positive
    assert track.check_alert(0.5, 3, 10.0, 0.0) is False  # Only 1 positive

    # Step 2: second frame -> positive
    results = tracker.update(det, current_time=1.0)
    track, _, _ = results[0]
    track.add_prediction(0.7)
    assert track.check_alert(0.5, 3, 10.0, 1.0) is False  # 2 positives

    # Step 3: third frame -> positive -> should fire alert!
    results = tracker.update(det, current_time=2.0)
    track, _, _ = results[0]
    track.add_prediction(0.9)
    assert track.check_alert(0.5, 3, 10.0, 2.0) is True  # 3 of 3 >= 0.5 -> Alert!

    # Step 4: fourth frame -> during cooldown -> should NOT fire
    results = tracker.update(det, current_time=3.0)
    track, _, _ = results[0]
    track.add_prediction(0.9)
    assert track.check_alert(0.5, 3, 10.0, 3.0) is False  # Cooldown active (3s < 10s)


def test_full_pipeline_synthetic():
    cfg = AppConfig()
    models_dir = Path(__file__).parent.parent.parent / "models"
    det_model = models_dir / cfg.DETECTOR_MODEL
    cls_model = models_dir / cfg.CLASSIFIER_MODEL

    assert det_model.exists(), f"Missing detector model at {det_model}"
    assert cls_model.exists(), f"Missing classifier model at {cls_model}"

    det_session = create_session(det_model, cfg)
    cls_session = create_session(cls_model, cfg)
    warmup_session(det_session)
    warmup_session(cls_session)

    detector = PersonDetector(det_session, cfg)
    classifier = SmokingClassifier(cls_session, cfg)
    pipeline = SmokingPipeline(detector, classifier, cfg)

    # Create synthetic frame with random content
    frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    res = pipeline.process_frame(frame, timestamp_s=0.0)

    assert "persons" in res
    assert "latencies" in res
    assert "new_alerts" in res
    print("\nSynthetic pipeline run successfully. Latencies:", res["latencies"])


if __name__ == "__main__":
    test_letterbox_unletterbox()
    test_iou_computation()
    test_tracker_temporal_voting()
    test_full_pipeline_synthetic()
    print("All tests passed!")
