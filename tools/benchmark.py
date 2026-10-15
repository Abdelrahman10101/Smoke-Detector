"""Performance and latency benchmark for smoking detection pipeline on CPU."""
import argparse
import sys
import time
from pathlib import Path

# Add project root and service root to sys.path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "service"))

import numpy as np
import onnxruntime as ort

from app.config import AppConfig
from app.onnx_session import create_session, warmup_session
from app.detector import PersonDetector
from app.classifier import SmokingClassifier
from app.pipeline import SmokingPipeline


def run_benchmark(n_frames: int = 50, threads: int = 4):
    cfg = AppConfig()
    cfg.ORT_THREADS = threads
    models_dir = Path("models")

    det_model = models_dir / cfg.DETECTOR_MODEL
    cls_model = models_dir / cfg.CLASSIFIER_MODEL

    print("=" * 60)
    print(f"BENCHMARK: Smoking Detection (CPU Threads: {threads})")
    print("=" * 60)

    det_sess = create_session(det_model, cfg)
    cls_sess = create_session(cls_model, cfg)
    warmup_session(det_sess)
    warmup_session(cls_sess)

    detector = PersonDetector(det_sess, cfg)
    classifier = SmokingClassifier(cls_sess, cfg)
    pipeline = SmokingPipeline(detector, classifier, cfg)

    # 1. Benchmark YOLO26n Detector
    dummy_frame = np.random.randint(0, 256, (720, 1280, 3), dtype=np.uint8)
    det_times = []
    for _ in range(n_frames):
        t0 = time.perf_counter()
        detector.detect(dummy_frame)
        det_times.append((time.perf_counter() - t0) * 1000.0)
    det_times = np.array(det_times)
    print(f"\n[1] YOLO26n (640x640 input, 720p frame):")
    print(f"    Mean Latency: {det_times.mean():.2f} ms")
    print(f"    p50 Latency:  {np.median(det_times):.2f} ms")
    print(f"    p95 Latency:  {np.percentile(det_times, 95):.2f} ms")
    print(f"    Max FPS:      {1000.0 / det_times.mean():.1f} FPS")

    # 2. Benchmark EfficientNet-B0 Classifier (single & batch)
    for batch_size in [1, 2, 4]:
        dummy_crops = [np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8) for _ in range(batch_size)]
        cls_times = []
        for _ in range(n_frames):
            t0 = time.perf_counter()
            classifier.classify(dummy_crops)
            cls_times.append((time.perf_counter() - t0) * 1000.0)
        cls_times = np.array(cls_times)
        print(f"\n[2] EfficientNet-B0 (Batch size: {batch_size}):")
        print(f"    Mean Latency: {cls_times.mean():.2f} ms ({cls_times.mean()/batch_size:.2f} ms/crop)")
        print(f"    p95 Latency:  {np.percentile(cls_times, 95):.2f} ms")

    # 3. Full Pipeline synthetic throughput
    pipe_times = []
    for i in range(n_frames):
        t0 = time.perf_counter()
        pipeline.process_frame(dummy_frame, timestamp_s=float(i) * 0.2)
        pipe_times.append((time.perf_counter() - t0) * 1000.0)
    pipe_times = np.array(pipe_times)
    print(f"\n[3] Full Pipeline (Detector + Tracker + Gating):")
    print(f"    Mean Latency: {pipe_times.mean():.2f} ms")
    print(f"    p95 Latency:  {np.percentile(pipe_times, 95):.2f} ms")
    print(f"    Sustained:    {1000.0 / pipe_times.mean():.1f} frames/sec")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    run_benchmark(args.frames, args.threads)
