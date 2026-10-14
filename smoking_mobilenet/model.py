"""Model builder (MobileNetV3-Large, EfficientNet-B0, EfficientNet-V2-S, etc.) and ONNX wrapper."""
import torch
import torch.nn as nn
from torchvision import models

from config import MEAN_255, STD_255, Config


def build_model(cfg: Config, pretrained: bool = True) -> nn.Module:
    arch = cfg.arch.lower()
    if arch == "mobilenet_v3_large":
        W = models.MobileNet_V3_Large_Weights
        weights = (W.IMAGENET1K_V2 if cfg.weights == "v2" else W.IMAGENET1K_V1) if pretrained else None
        m = models.mobilenet_v3_large(weights=weights)
        m.classifier[2] = nn.Dropout(p=cfg.dropout, inplace=True)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, 1)
        return m
    elif arch.startswith("efficientnet"):
        if not hasattr(models, arch):
            raise ValueError(
                f"Unknown architecture '{arch}' in torchvision.models. "
                "Supported examples: efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_v2_s"
            )
        fn = getattr(models, arch)
        m = fn(weights="DEFAULT" if pretrained else None)
        in_f = m.classifier[1].in_features
        m.classifier = nn.Sequential(
            nn.Dropout(p=cfg.dropout, inplace=True),
            nn.Linear(in_f, 1),
        )
        return m
    else:
        raise ValueError(f"Unsupported architecture: {cfg.arch}")


def set_backbone_trainable(model: nn.Module, trainable: bool):
    for p in model.features.parameters():
        p.requires_grad = trainable


def param_groups(model: nn.Module, weight_decay: float):
    """No weight decay on BatchNorm / bias parameters."""
    decay, no_decay = [], []
    for p in model.parameters():
        (no_decay if p.ndim <= 1 else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


class ExportWrapper(nn.Module):
    """Input: float32 NCHW RGB in 0-255 (letterboxed). Output: P(smoking) in [0, 1].

    ImageNet normalisation is baked in so TensorRT / ONNX Runtime need no extra preprocessing.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(MEAN_255).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(STD_255).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        return torch.sigmoid(self.model(x))


def load_checkpoint_model(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = Config(**ckpt["cfg"]) if "cfg" in ckpt else Config()
    model = build_model(cfg, pretrained=False)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), cfg, ckpt
