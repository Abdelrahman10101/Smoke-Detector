"""Central configuration for the smoking / not-smoking classifier."""
import os
from dataclasses import asdict, dataclass
from pathlib import Path

DATA_URL = "https://data.mendeley.com/public-api/zip/7b52hhzs3r/download/1"

# All paths are relative to the current working directory unless overridden by env vars.
DATA_DIR = Path(os.environ.get("SMOKING_DATA_DIR", "data")).resolve()
OUT_DIR = Path(os.environ.get("SMOKING_OUT_DIR", "outputs")).resolve()
CKPT_DIR = OUT_DIR / "checkpoints"

TRAIN_SPLIT = "training_data"
VAL_SPLIT = "validation_data"
# testing_data/ is intentionally ignored.

CLASS_NAMES = ["notsmoking", "smoking"]  # index == label
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ImageNet statistics in 0-255 range (used on GPU and baked into the ONNX model).
MEAN_255 = (123.675, 116.28, 103.53)
STD_255 = (58.395, 57.12, 57.375)


@dataclass
class Config:
    # model
    # Supported architectures: mobilenet_v3_large, efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_v2_s
    arch: str = "efficientnet_b0"
    weights: str = "v2"            # for mobilenet: v2/v1; for efficientnet: default ImageNet
    dropout: float = 0.3
    img_size: int = 224
    pad_value: int = 114           # letterbox padding colour (same in train/val/inference)
    max_side: int = 640            # images are shrunk to this long side before processing (speed)

    # optimisation
    epochs: int = 100
    batch_size: int = 64
    lr: float = 1e-3               # must be > min_lr, otherwise ReduceLR never has an effect
    weight_decay: float = 1e-2     # AdamW
    warmup_epochs: int = 2         # backbone frozen, scheduler & early stopping inactive

    # ReduceLROnPlateau
    sched_factor: float = 0.5
    sched_patience: int = 2
    min_lr: float = 2.5e-4

    # EarlyStopping
    es_patience: int = 5
    es_min_delta: float = 1e-4

    # data / augmentation
    cctv_prob: float = 0.8         # probability that a training image is CCTV-degraded
    num_workers: int = 2
    seed: int = 42
    leak_hamming: int = 4          # dHash distance treated as near-duplicate (audit_data.py)

    def to_dict(self):
        return asdict(self)


def resolve_data_root() -> Path:
    """Dataset root = folder that contains training_data/ and validation_data/."""
    env = os.environ.get("SMOKING_DATA_ROOT")
    if env:
        return Path(env)
    marker = DATA_DIR / "dataset_root.txt"
    if marker.exists():
        return Path(marker.read_text().strip())
    raise FileNotFoundError(
        "Dataset not found. Run `python download_data.py` first or set SMOKING_DATA_ROOT."
    )
