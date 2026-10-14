"""Training / evaluation helpers shared by train.py and evaluate.py."""
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)

from config import MEAN_255, STD_255


def normalize(x_uint8: torch.Tensor) -> torch.Tensor:
    """uint8 NCHW (0-255) -> ImageNet-normalised float, on whatever device x lives on."""
    mean = torch.tensor(MEAN_255, device=x_uint8.device).view(1, 3, 1, 1)
    std = torch.tensor(STD_255, device=x_uint8.device).view(1, 3, 1, 1)
    return (x_uint8.float() - mean) / std


class EarlyStopping:
    """Stops after `patience` consecutive epochs without val-loss improvement > min_delta."""

    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience, self.min_delta = patience, min_delta
        self.best, self.counter, self.should_stop = float("inf"), 0, False

    def step(self, value: float, active: bool = True) -> bool:
        """Returns True when `value` is a new best. `active=False` (warm-up) never counts patience."""
        if value < self.best - self.min_delta:
            self.best, self.counter = value, 0
            return True
        if active:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return False

    def state_dict(self):
        return {"best": self.best, "counter": self.counter, "should_stop": self.should_stop}

    def load_state_dict(self, d):
        self.best, self.counter, self.should_stop = d["best"], d["counter"], d["should_stop"]


def train_one_epoch(model, loader, optimizer, scaler, criterion, device, amp, backbone_frozen):
    model.train()
    if backbone_frozen:
        model.features.eval()          # keep BatchNorm statistics fixed during warm-up
    total, loss_sum, correct = 0, 0.0, 0
    for x, y in loader:
        x = normalize(x.to(device, non_blocking=True))
        y = y.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits = model(x).squeeze(1)
            loss = criterion(logits, y)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        n = y.size(0)
        total += n
        loss_sum += loss.item() * n
        correct += ((logits.detach() > 0).float() == y).sum().item()
    return {"loss": loss_sum / total, "acc": correct / total}


@torch.no_grad()
def predict_loader(model, loader, device, criterion=None):
    """Returns dict(loss, labels, probs)."""
    model.eval()
    probs, labels, loss_sum, total = [], [], 0.0, 0
    for x, y in loader:
        x = normalize(x.to(device, non_blocking=True))
        y = y.to(device)
        logits = model(x).squeeze(1).float()
        if criterion is not None:
            loss_sum += criterion(logits, y).item() * y.size(0)
        total += y.size(0)
        probs.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(y.cpu().numpy())
    return {"loss": loss_sum / max(total, 1), "labels": np.concatenate(labels).astype(int),
            "probs": np.concatenate(probs)}


def compute_metrics(labels, probs, thr=0.5):
    pred = (probs >= thr).astype(int)
    try:
        auc = float(roc_auc_score(labels, probs))
    except ValueError:
        auc = float("nan")
    return {
        "threshold": float(thr),
        "accuracy": float(accuracy_score(labels, pred)),
        "precision": float(precision_score(labels, pred, zero_division=0)),
        "recall": float(recall_score(labels, pred, zero_division=0)),
        "f1": float(f1_score(labels, pred, zero_division=0)),
        "auc": auc,
        "confusion_matrix": confusion_matrix(labels, pred, labels=[0, 1]).tolist(),
    }
