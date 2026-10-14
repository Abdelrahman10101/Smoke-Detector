"""Evaluate best.pt on the validation set (CCTV-degraded AND clean), tune the threshold,
and save confusion matrices + a gallery of misclassified images.

    python evaluate.py [--ckpt outputs/checkpoints/best.pt] [--recall-target 0.95]
"""
import argparse
import json

import matplotlib
import numpy as np
import torch
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from config import CKPT_DIR, CLASS_NAMES, OUT_DIR, VAL_SPLIT, resolve_data_root  # noqa: E402
from dataset import SmokingDataset, list_samples  # noqa: E402
from engine import compute_metrics, predict_loader  # noqa: E402
from model import load_checkpoint_model  # noqa: E402


def tune_thresholds(labels, probs, recall_target):
    grid = np.linspace(0.01, 0.99, 99)
    f1s = [compute_metrics(labels, probs, t)["f1"] for t in grid]
    t_f1 = float(grid[int(np.argmax(f1s))])
    ok = [t for t in grid if compute_metrics(labels, probs, t)["recall"] >= recall_target]
    t_rec = float(max(ok)) if ok else None      # highest threshold that still meets the recall target
    return {"f1_optimal": t_f1, f"recall>={recall_target}": t_rec}


def plot_cm(cm, title, path):
    plt.figure(figsize=(3.8, 3.4))
    plt.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i][j], ha="center", va="center",
                     color="white" if cm[i][j] > np.max(cm) / 2 else "black")
    plt.xticks([0, 1], CLASS_NAMES)
    plt.yticks([0, 1], CLASS_NAMES)
    plt.xlabel("predicted")
    plt.ylabel("true")
    plt.title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def error_gallery(ds, labels, probs, thr, path, max_n=24):
    wrong = np.where((probs >= thr).astype(int) != labels)[0]
    if len(wrong) == 0:
        return
    wrong = wrong[np.argsort(-np.abs(probs[wrong] - 0.5))][:max_n]   # most confident mistakes first
    cols = 6
    rows = int(np.ceil(len(wrong) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.5 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, i in zip(axes.ravel(), wrong):
        ax.imshow(ds.cache[i])
        ax.set_title(f"true={CLASS_NAMES[labels[i]]}\np(smoke)={probs[i]:.2f}", fontsize=7)
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(CKPT_DIR / "best.pt"))
    ap.add_argument("--recall-target", type=float, default=0.95)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, ck = load_checkpoint_model(args.ckpt, device)
    print(f"Loaded {args.ckpt} (epoch {ck.get('epoch')})")
    paths, labels = list_samples(resolve_data_root(), VAL_SPLIT, use_exclusions=False)

    results, datasets = {}, {}
    for name, degrade in [("cctv", True), ("clean", False)]:
        ds = SmokingDataset(paths, labels, cfg, train=False, degrade=degrade)
        datasets[name] = ds
        out = predict_loader(model, DataLoader(ds, batch_size=args.batch_size), device)
        results[name] = out

    thresholds = tune_thresholds(results["cctv"]["labels"], results["cctv"]["probs"], args.recall_target)
    chosen = thresholds["f1_optimal"]
    summary = {"thresholds": thresholds, "chosen_threshold": chosen}
    for name, out in results.items():
        summary[name] = {"@0.5": compute_metrics(out["labels"], out["probs"], 0.5),
                         "@f1_optimal": compute_metrics(out["labels"], out["probs"], chosen)}
        plot_cm(summary[name]["@0.5"]["confusion_matrix"], f"{name} val @0.5", OUT_DIR / f"cm_{name}.png")
        m = summary[name]["@0.5"]
        print(f"[{name:5s}] @0.5: acc {m['accuracy']:.3f} P {m['precision']:.3f} R {m['recall']:.3f} "
              f"F1 {m['f1']:.3f} AUC {m['auc']:.3f}")
    print("thresholds (tuned on CCTV val):", thresholds)
    error_gallery(datasets["cctv"], results["cctv"]["labels"], results["cctv"]["probs"], 0.5,
                  OUT_DIR / "errors_cctv.png")

    (OUT_DIR / "metrics.json").write_text(json.dumps(summary, indent=2))
    print("saved", OUT_DIR / "metrics.json")


if __name__ == "__main__":
    main()
