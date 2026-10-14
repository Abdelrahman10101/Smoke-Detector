"""EfficientNet-B0 smoking classifier wrapper."""
import time
from typing import List, Tuple
import numpy as np
import onnxruntime as ort

from .config import AppConfig
from .preprocess import prepare_classifier_input


class SmokingClassifier:
    def __init__(self, session: ort.InferenceSession, cfg: AppConfig):
        self.session = session
        self.cfg = cfg
        self.input_name = session.get_inputs()[0].name
        self.output_name = session.get_outputs()[0].name

    def classify(self, rgb_crops: List[np.ndarray]) -> Tuple[List[float], float]:
        """Classify a list of RGB crops as smoking vs not-smoking.
        
        Args:
            rgb_crops: List of RGB numpy arrays (H, W, 3)
            
        Returns:
            probs: List of P(smoking) probabilities in [0.0, 1.0]
            latency_ms: Total classification latency in milliseconds
        """
        if not rgb_crops:
            return [], 0.0

        t0 = time.perf_counter()
        
        # Preprocess each crop (letterbox to 224x224, float32, range 0-255)
        prepped = [prepare_classifier_input(crop, imgsz=224) for crop in rgb_crops]
        batch_input = np.stack(prepped, axis=0)  # Shape: (N, 3, 224, 224)
        
        outputs = self.session.run([self.output_name], {self.input_name: batch_input})
        raw_probs = outputs[0]  # Shape: (N, 1)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        
        probs = [float(p[0]) for p in raw_probs]
        return probs, latency_ms
