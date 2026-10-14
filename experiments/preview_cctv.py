"""Save a preview grid: clean letterbox vs. CCTV-degraded vs. a random training sample.

Look at outputs/preview_cctv.png BEFORE training to make sure the cigarette is still visible.
"""
import argparse

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from cctv_transforms import cctv_degrade, letterbox  # noqa: E402
from config import OUT_DIR, TRAIN_SPLIT, Config, resolve_data_root  # noqa: E402
from dataset import SmokingDataset, list_samples, load_rgb  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8, help="images per class")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = Config()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths, labels = list_samples(resolve_data_root(), TRAIN_SPLIT)
    rng = np.random.default_rng(args.seed)
    pick = [int(i) for c in (1, 0) for i in rng.choice(np.where(np.array(labels) == c)[0], args.n, replace=False)]
    ds = SmokingDataset([paths[i] for i in pick], [labels[i] for i in pick], cfg, train=True)

    rows = ["clean (letterbox)", "CCTV (eval style)", "train sample"]
    cols = len(pick)
    per_fig = 8
    for part, start in enumerate(range(0, cols, per_fig)):
        idx = list(range(start, min(start + per_fig, cols)))
        fig, axes = plt.subplots(3, len(idx), figsize=(2.2 * len(idx), 7), squeeze=False)
        for c, i in enumerate(idx):
            img = load_rgb(paths[pick[i]], cfg.max_side)
            tiles = [letterbox(img, cfg.img_size, cfg.pad_value),
                     letterbox(cctv_degrade(img, rng), cfg.img_size, cfg.pad_value),
                     ds.render_train(i)]
            for r in range(3):
                axes[r, c].imshow(tiles[r])
                axes[r, c].axis("off")
                if r == 0:
                    axes[r, c].set_title("smoking" if labels[pick[i]] else "notsmoking", fontsize=9)
        for r, name in enumerate(rows):
            axes[r, 0].text(-0.05, 0.5, name, transform=axes[r, 0].transAxes, rotation=90,
                            va="center", ha="right", fontsize=9)
        plt.tight_layout()
        out = OUT_DIR / f"preview_cctv_{part}.png"
        plt.savefig(out, dpi=110)
        plt.close()
        print("saved", out)


if __name__ == "__main__":
    main()
