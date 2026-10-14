"""Train Smoking vs Not-Smoking Classifier with CCTV-style augmentation.

Supports MobileNetV3-Large, EfficientNet-B0, EfficientNet-V2-S, etc.
AdamW | ReduceLROnPlateau(min_lr=2.5e-4) | EarlyStopping(patience=5) | up to 100 epochs.

    python train.py --arch efficientnet_b0
    python train.py --arch mobilenet_v3_large
"""
import argparse
import csv
import random
import time

import matplotlib
import numpy as np
import torch
import torch.nn as nn

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from config import CKPT_DIR, OUT_DIR, Config  # noqa: E402
from dataset import get_loaders  # noqa: E402
from engine import (EarlyStopping, compute_metrics, predict_loader,  # noqa: E402
                    train_one_epoch)
from model import build_model, param_groups, set_backbone_trainable  # noqa: E402


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_history(history):
    if not history:
        return
    with open(OUT_DIR / "history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)


def plot_history(history):
    if not history:
        return
    ep = [h["epoch"] for h in history]
    for name, keys, ylabel in [("loss.png", ("train_loss", "val_loss"), "loss"),
                               ("accuracy.png", ("train_acc", "val_acc"), "accuracy"),
                               ("lr.png", ("lr",), "learning rate")]:
        plt.figure(figsize=(6, 4))
        for k in keys:
            plt.plot(ep, [h[k] for h in history], marker="o", ms=3, label=k)
        plt.xlabel("epoch")
        plt.ylabel(ylabel)
        if name == "lr.png":
            plt.yscale("log")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUT_DIR / name, dpi=120)
        plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", type=str, default=None,
                    help="Model architecture: mobilenet_v3_large, efficientnet_b0, efficientnet_v2_s, etc.")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--min-lr", type=float)
    ap.add_argument("--img-size", type=int)
    ap.add_argument("--warmup-epochs", type=int)
    ap.add_argument("--es-patience", type=int, help="Early stopping patience")
    ap.add_argument("--workers", type=int)
    ap.add_argument("--limit", type=int, help="use only N images per split (debug)")
    ap.add_argument("--smoke", action="store_true", help="tiny run to verify the pipeline")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-pretrained", action="store_true")
    args = ap.parse_args()

    cfg = Config()
    for k, v in [("arch", args.arch), ("epochs", args.epochs), ("batch_size", args.batch_size),
                 ("lr", args.lr), ("min_lr", args.min_lr), ("img_size", args.img_size),
                 ("warmup_epochs", args.warmup_epochs), ("es_patience", args.es_patience),
                 ("num_workers", args.workers)]:
        if v is not None:
            setattr(cfg, k, v)
    if args.smoke:
        cfg.epochs, cfg.warmup_epochs, args.limit = 2, 1, args.limit or 64
        cfg.batch_size = min(cfg.batch_size, 16)
    assert cfg.lr > cfg.min_lr, "lr must be above min_lr or ReduceLROnPlateau can never act"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    print(f"device: {device} | AMP: {amp}")
    print(cfg)

    train_loader, val_loader = get_loaders(cfg, limit=args.limit)
    model = build_model(cfg, pretrained=not args.no_pretrained).to(device)
    optimizer = torch.optim.AdamW(param_groups(model, cfg.weight_decay), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=cfg.sched_factor, patience=cfg.sched_patience,
        threshold=cfg.es_min_delta, threshold_mode="abs", min_lr=cfg.min_lr)
    es = EarlyStopping(cfg.es_patience, cfg.es_min_delta)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    criterion = nn.BCEWithLogitsLoss()

    history, start_epoch = [], 1
    last_path, best_path = CKPT_DIR / "last.pt", CKPT_DIR / "best.pt"
    if args.resume and last_path.exists():
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        es.load_state_dict(ck["es"])
        history, start_epoch = ck["history"], ck["epoch"] + 1
        print(f"Resumed from epoch {ck['epoch']} (best val loss {es.best:.4f})")

    t_start = time.time()
    try:
        for epoch in range(start_epoch, cfg.epochs + 1):
            frozen = epoch <= cfg.warmup_epochs
            set_backbone_trainable(model, not frozen)
            lr = optimizer.param_groups[0]["lr"]
            t0 = time.time()

            tr = train_one_epoch(model, train_loader, optimizer, scaler, criterion,
                                 device, amp, frozen)
            va = predict_loader(model, val_loader, device, criterion)
            m = compute_metrics(va["labels"], va["probs"], 0.5)

            if not frozen:
                scheduler.step(va["loss"])
            improved = es.step(va["loss"], active=not frozen)
            if improved:
                torch.save({"model": model.state_dict(), "epoch": epoch, "val_loss": va["loss"],
                            "cfg": cfg.to_dict()}, best_path)

            history.append({"epoch": epoch, "lr": lr, "train_loss": tr["loss"],
                            "train_acc": tr["acc"], "val_loss": va["loss"], "val_acc": m["accuracy"],
                            "val_f1": m["f1"], "val_auc": m["auc"], "seconds": time.time() - t0})
            save_history(history)
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                        "es": es.state_dict(), "epoch": epoch, "history": history,
                        "cfg": cfg.to_dict()}, last_path)

            new_lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch:3d}/{cfg.epochs} | lr {lr:.2e} | train loss {tr['loss']:.4f} "
                  f"acc {tr['acc']:.3f} | val loss {va['loss']:.4f} acc {m['accuracy']:.3f} "
                  f"f1 {m['f1']:.3f} auc {m['auc']:.3f} | {time.time() - t0:.0f}s"
                  f"{' *best*' if improved else ''}"
                  f"{' [warm-up]' if frozen else f' [patience {es.counter}/{cfg.es_patience}]'}")
            if new_lr < lr:
                print(f"    ReduceLROnPlateau: lr {lr:.2e} -> {new_lr:.2e}")
            if es.should_stop:
                print(f"Early stopping at epoch {epoch} (no val-loss improvement for "
                      f"{cfg.es_patience} epochs).")
                break
    except KeyboardInterrupt:
        print("Interrupted - last.pt is saved, use --resume to continue.")

    plot_history(history)
    if best_path.exists():
        ck = torch.load(best_path, map_location="cpu", weights_only=False)
        print(f"Best epoch {ck['epoch']} | val loss {ck['val_loss']:.4f} | {best_path}")
    print(f"Total time: {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
