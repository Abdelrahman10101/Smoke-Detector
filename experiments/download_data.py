"""Download the Mendeley dataset, unzip/unrar it, and locate the dataset root."""
import argparse
import os
import shutil
import subprocess
import tarfile
import urllib.request
import zipfile
from pathlib import Path

from config import DATA_DIR, DATA_URL, TRAIN_SPLIT, VAL_SPLIT


def norm(name: str) -> str:
    return name.lower().replace("_", "").replace("-", "").replace(" ", "")


def find_dataset_root(base: Path):
    """Return the folder that has both the training and validation split folders."""
    want = {norm(TRAIN_SPLIT), norm(VAL_SPLIT)}
    for d in [base, *[p for p in base.rglob("*") if p.is_dir()]]:
        children = {norm(c.name) for c in d.iterdir() if c.is_dir()}
        if want <= children:
            return d
    return None


def download(url: str, dst: Path):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as r, open(dst, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e6:8.1f} / {total / 1e6:.1f} MB", end="")
            else:
                print(f"\r  {done / 1e6:8.1f} MB", end="")
    print()


def extract_single_archive(archive_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = archive_path.suffix.lower()
    if ext == ".zip":
        with zipfile.ZipFile(archive_path) as z:
            z.extractall(out_dir)
    elif ext in {".rar", ".cbr"}:
        print(f"Extracting RAR archive: {archive_path.name} ...")
        if shutil.which("unrar"):
            subprocess.run(["unrar", "x", "-o+", str(archive_path), str(out_dir)], check=True)
        elif shutil.which("7z"):
            subprocess.run(["7z", "x", f"-o{out_dir}", "-y", str(archive_path)], check=True)
        else:
            raise RuntimeError(
                f"Found RAR archive '{archive_path.name}' but neither 'unrar' nor '7z' command was found.\n"
                "In Colab, install unrar with:\n  !apt-get install -y unrar"
            )
    elif ext in {".tar", ".gz", ".tgz", ".bz2"}:
        with tarfile.open(archive_path) as t:
            t.extractall(out_dir)


def extract_recursive(archive_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    extract_single_archive(archive_path, out_dir)
    # Mendeley zip files frequently contain an inner .rar or .zip file:
    archive_exts = {".zip", ".rar", ".tar", ".gz", ".tgz"}
    seen = {archive_path.resolve()}
    while True:
        inner = [
            p for p in out_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in archive_exts and p.resolve() not in seen
        ]
        if not inner:
            break
        for p in inner:
            seen.add(p.resolve())
            target = p.with_suffix("")
            extract_single_archive(p, target)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=DATA_URL)
    ap.add_argument("--force", action="store_true", help="re-download even if already present")
    args = ap.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw = DATA_DIR / "raw"
    zip_path = DATA_DIR / "smoking.zip"

    root = find_dataset_root(raw) if raw.exists() and not args.force else None
    if root is None:
        if args.force and raw.exists():
            shutil.rmtree(raw)
        if not zip_path.exists() or args.force:
            print(f"Downloading {args.url}")
            download(args.url, zip_path)
        print("Extracting ...")
        extract_recursive(zip_path, raw)
        root = find_dataset_root(raw)
    if root is None:
        raise SystemExit(f"Could not find {TRAIN_SPLIT}/{VAL_SPLIT} inside {raw}. Inspect the archive.")

    (DATA_DIR / "dataset_root.txt").write_text(str(root))
    print(f"Dataset root: {root}")
    for c in sorted(root.iterdir()):
        if c.is_dir():
            n = sum(1 for p in c.rglob("*") if p.is_file())
            print(f"  {c.name}: {n} files")


if __name__ == "__main__":
    main()
