"""Datasets and data loaders (train: random CCTV + crop-free aug; val: fixed-seed CCTV)."""
import zlib
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset

from cctv_transforms import (cap_long_side, cctv_degrade, letterbox, random_affine_nocrop,
                             random_flip)
from config import (CLASS_NAMES, IMG_EXT, OUT_DIR, TRAIN_SPLIT, VAL_SPLIT, Config,
                    resolve_data_root)


# --------------------------------------------------------------------------------------
# File listing
# --------------------------------------------------------------------------------------
def _norm(name):
    return name.lower().replace("_", "").replace("-", "").replace(" ", "")


def find_child(parent: Path, name: str) -> Path:
    for c in parent.iterdir():
        if c.is_dir() and _norm(c.name) == _norm(name):
            return c
    raise FileNotFoundError(f"'{name}' not found in {parent}")


def load_exclusions(root: Path):
    f = OUT_DIR / "exclude.txt"
    if not f.exists():
        return set()
    return {ln.strip() for ln in f.read_text().splitlines() if ln.strip()}


def list_samples(root: Path, split: str, use_exclusions=True):
    """Return (paths, labels) for a split. Label 1 = smoking, 0 = notsmoking."""
    split_dir = find_child(root, split)
    exclude = load_exclusions(root) if use_exclusions else set()
    paths, labels = [], []
    for label, cname in enumerate(CLASS_NAMES):
        for p in sorted(find_child(split_dir, cname).rglob("*")):
            if p.is_file() and p.suffix.lower() in IMG_EXT:
                if p.relative_to(root).as_posix() in exclude:
                    continue
                paths.append(p)
                labels.append(label)
    return paths, labels


def load_rgb(path, max_side=640) -> np.ndarray:
    """Fast RGB loader: JPEG draft mode + EXIF orientation + long-side cap."""
    with Image.open(path) as im:
        im.draft("RGB", (max_side, max_side))
        im = ImageOps.exif_transpose(im)
        arr = np.array(im.convert("RGB"))
    return cap_long_side(arr, max_side)


# --------------------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------------------
class SmokingDataset(Dataset):
    """Returns (uint8 CHW tensor, float label). Normalisation is done on the GPU (engine.normalize)."""

    def __init__(self, paths, labels, cfg: Config, train: bool, degrade: bool = True):
        self.paths, self.labels, self.cfg = list(paths), list(labels), cfg
        self.train, self.degrade = train, degrade
        self._rng, self._rng_pid = None, None
        self.cache = None
        if not train:  # deterministic -> pre-render once
            print(f"  pre-rendering {len(self.paths)} eval images (cctv={degrade}) ...")
            self.cache = np.stack([self._render_eval(p) for p in self.paths])

    # ---- rendering ----
    def _render_eval(self, path: Path):
        img = load_rgb(path, self.cfg.max_side)
        if self.degrade:
            rng = np.random.default_rng(zlib.crc32(path.name.encode()) ^ self.cfg.seed)
            img = cctv_degrade(img, rng)
        return letterbox(img, self.cfg.img_size, self.cfg.pad_value)

    def _get_rng(self):
        import os
        pid = os.getpid()
        if self._rng is None or self._rng_pid != pid:   # fresh stream per worker process
            seed = (torch.initial_seed() + pid) % (2 ** 32)
            self._rng, self._rng_pid = np.random.default_rng(seed), pid
        return self._rng

    def render_train(self, i):
        """flip -> CCTV degradation (p=cctv_prob) -> crop-free affine -> letterbox."""
        cfg, rng = self.cfg, self._get_rng()
        img = load_rgb(self.paths[i], cfg.max_side)
        img = random_flip(img, rng)
        if rng.random() < cfg.cctv_prob:
            img = cctv_degrade(img, rng)
        if rng.random() < 0.7:
            img = random_affine_nocrop(img, rng, pad=cfg.pad_value)
        return letterbox(img, cfg.img_size, cfg.pad_value)

    # ---- torch API ----
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        arr = self.cache[i] if self.cache is not None else self.render_train(i)
        x = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1)
        return x, torch.tensor(float(self.labels[i]))


def _subset(paths, labels, n, seed):
    if n is None or n >= len(paths):
        return paths, labels
    idx = np.random.default_rng(seed).permutation(len(paths))[:n]
    return [paths[i] for i in idx], [labels[i] for i in idx]


def get_datasets(cfg: Config, limit=None):
    root = resolve_data_root()
    tp, tl = _subset(*list_samples(root, TRAIN_SPLIT), limit, cfg.seed)
    vp, vl = _subset(*list_samples(root, VAL_SPLIT), limit, cfg.seed)
    print(f"train: {len(tp)} images ({sum(tl)} smoking) | val: {len(vp)} images ({sum(vl)} smoking)")
    train_ds = SmokingDataset(tp, tl, cfg, train=True)
    val_ds = SmokingDataset(vp, vl, cfg, train=False, degrade=True)   # CCTV, fixed seed
    return train_ds, val_ds


def get_loaders(cfg: Config, limit=None):
    train_ds, val_ds = get_datasets(cfg, limit)
    pin = torch.cuda.is_available()
    w = cfg.num_workers
    kw = dict(num_workers=w, pin_memory=pin, persistent_workers=w > 0)
    if w > 0:
        kw["prefetch_factor"] = 4
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True, **kw)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0, pin_memory=pin)
    return train_loader, val_loader
