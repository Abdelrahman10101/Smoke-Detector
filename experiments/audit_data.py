"""Dataset audit: corrupt files, class balance, image sizes and train<->val leakage (dHash).

Writes outputs/audit_report.json and outputs/exclude.txt (leaking / corrupt TRAIN images
that dataset.py will skip).
"""
import json

import cv2
import numpy as np
from PIL import Image

from config import CLASS_NAMES, OUT_DIR, TRAIN_SPLIT, VAL_SPLIT, Config, resolve_data_root
from dataset import list_samples, load_rgb


def dhash(img_rgb, size=8):
    g = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    g = cv2.resize(g, (size + 1, size), interpolation=cv2.INTER_AREA)
    return (g[:, 1:] > g[:, :-1]).flatten()


def main():
    cfg = Config()
    root = resolve_data_root()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report, hashes, bad = {}, {}, []

    for split in (TRAIN_SPLIT, VAL_SPLIT):
        paths, labels = list_samples(root, split, use_exclusions=False)
        sizes, hs = [], []
        for p in paths:
            try:
                with Image.open(p) as im:
                    sizes.append(im.size)
                hs.append(dhash(load_rgb(p, cfg.max_side)))
            except Exception as e:  # corrupt / unreadable
                bad.append(p)
                hs.append(None)
                print(f"  BAD FILE {p}: {e}")
        hashes[split] = (paths, labels, hs)
        long_side = np.array([max(s) for s in sizes])
        report[split] = {
            "images": len(paths),
            **{CLASS_NAMES[c]: int(sum(1 for l in labels if l == c)) for c in (0, 1)},
            "long_side_min": int(long_side.min()), "long_side_median": int(np.median(long_side)),
            "long_side_max": int(long_side.max()),
            "share_long_side_lt_320": float((long_side < 320).mean()),
        }
        print(split, report[split])

    # leakage: near-duplicates between train and validation
    tp, tl, th = hashes[TRAIN_SPLIT]
    vp, vl, vh = hashes[VAL_SPLIT]
    valid_v = [(p, l, h) for p, l, h in zip(vp, vl, vh) if h is not None]
    V = np.stack([h for _, _, h in valid_v]) if valid_v else np.zeros((0, 64), bool)
    leaks, label_conflicts = [], []
    for p, l, h in zip(tp, tl, th):
        if h is None or len(V) == 0:
            continue
        d = (V != h[None]).sum(1)
        j = int(d.argmin())
        if d[j] <= cfg.leak_hamming:
            leaks.append(p)
            if valid_v[j][1] != l:
                label_conflicts.append((p.name, valid_v[j][0].name))
    print(f"\nTrain images that are near-duplicates of validation images: {len(leaks)}")
    print(f"Near-duplicates with DIFFERENT labels (label noise): {len(label_conflicts)}")

    exclude = {p.relative_to(root).as_posix() for p in leaks + [b for b in bad if TRAIN_SPLIT in str(b)]}
    (OUT_DIR / "exclude.txt").write_text("\n".join(sorted(exclude)))
    report.update({"corrupt_files": [str(b) for b in bad], "train_val_leaks": len(leaks),
                   "label_conflicts": label_conflicts, "excluded_from_train": len(exclude)})
    (OUT_DIR / "audit_report.json").write_text(json.dumps(report, indent=2))
    print(f"Excluded from training: {len(exclude)} -> {OUT_DIR / 'exclude.txt'}")


if __name__ == "__main__":
    main()
