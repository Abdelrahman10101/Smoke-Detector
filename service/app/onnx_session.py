"""ONNX Runtime session manager with warmup, provider checking, and thread configuration."""
import hashlib
import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import onnxruntime as ort

from .config import AppConfig

logger = logging.getLogger(__name__)


def verify_manifest(models_dir: Path, manifest_name: str = "manifest.json") -> bool:
    manifest_path = models_dir / manifest_name
    if not manifest_path.exists():
        logger.warning(f"Manifest not found at {manifest_path}, skipping checksum verification.")
        return True
    
    try:
        manifest = json.loads(manifest_path.read_text())
        for model_file, info in manifest.get("models", {}).items():
            model_path = models_dir / model_file
            if not model_path.exists():
                logger.error(f"Model file {model_path} specified in manifest does not exist!")
                return False
            expected_sha = info.get("sha256")
            actual_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
            if expected_sha and expected_sha != actual_sha:
                logger.error(
                    f"Checksum mismatch for {model_file}: expected {expected_sha}, got {actual_sha}"
                )
                return False
        logger.info(f"Manifest checksums verified successfully from {manifest_path}")
        return True
    except Exception as e:
        logger.error(f"Error checking manifest {manifest_path}: {e}")
        return False


def create_session(
    model_path: Path,
    cfg: AppConfig,
    device_id: int = 0
) -> ort.InferenceSession:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = max(1, cfg.ORT_THREADS)
    opts.inter_op_num_threads = 1
    opts.log_severity_level = 3  # Errors only

    avail = ort.get_available_providers()
    providers: List[Tuple[str, dict] | str] = []

    if cfg.REQUIRE_GPU:
        if "CUDAExecutionProvider" not in avail:
            raise RuntimeError(
                "REQUIRE_GPU=True, but CUDAExecutionProvider is not available in onnxruntime!"
            )
        providers = [
            ("CUDAExecutionProvider", {
                "device_id": device_id,
                "cudnn_conv_algo_search": "EXHAUSTIVE"
            }),
            "CPUExecutionProvider"
        ]
    else:
        # Default CPU execution or fallback
        providers = ["CPUExecutionProvider"]

    session = ort.InferenceSession(str(model_path), sess_options=opts, providers=providers)
    active_providers = session.get_providers()
    logger.info(f"Loaded {model_path.name} with providers: {active_providers}")

    if cfg.REQUIRE_GPU and "CUDAExecutionProvider" not in active_providers:
        raise RuntimeError(
            f"CUDAExecutionProvider failed to activate for {model_path.name}. "
            f"Active providers: {active_providers}"
        )

    return session


def warmup_session(session: ort.InferenceSession, default_batch: int = 1, iterations: int = 3):
    """Run dummy inferences through the model to prime memory allocation and graph optimizations."""
    input_meta = session.get_inputs()[0]
    input_name = input_meta.name
    shape = list(input_meta.shape)

    # Resolve dynamic dimensions
    resolved_shape = []
    for i, dim in enumerate(shape):
        if isinstance(dim, str) or dim is None or dim <= 0:
            resolved_shape.append(default_batch if i == 0 else 224)
        else:
            resolved_shape.append(dim)

    # Cast to appropriate dtype
    dtype = np.float32 if "float" in input_meta.type else np.uint8
    dummy_input = np.random.randint(0, 256, size=resolved_shape).astype(dtype)

    for _ in range(iterations):
        session.run(None, {input_name: dummy_input})
    logger.info(f"Warmed up session for input {input_name} with shape {resolved_shape}")
