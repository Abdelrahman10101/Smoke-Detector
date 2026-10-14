"""ONNX Runtime inference (reference for the TensorRT deployment preprocessing).

    python predict.py --model outputs/smoking_mobilenetv3.onnx IMAGE_OR_FOLDER [...]
    python predict.py --model ... --cctv IMAGE        # degrade the image like a CCTV frame first

Only needs numpy, opencv, onnxruntime and cctv_transforms.py (for letterbox).
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from cctv_transforms import cctv_degrade, letterbox

EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect(inputs):
    files = []
    for s in inputs:
        p = Path(s)
        files += sorted(f for f in p.rglob("*") if f.suffix.lower() in EXT) if p.is_dir() else [p]
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--model", default="outputs/smoking_mobilenetv3.onnx")
    ap.add_argument("--meta", default=None, help="model_meta.json (default: next to the model)")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--cctv", action="store_true", help="apply CCTV degradation before inference")
    ap.add_argument("--save-dir", default=None, help="save the exact network input images here")
    args = ap.parse_args()

    meta_path = Path(args.meta) if args.meta else Path(args.model).with_name("model_meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"threshold": 0.5, "input_shape": ["N", 3, 224, 224]}
    thr = args.threshold if args.threshold is not None else meta["threshold"]
    size = meta["input_shape"][-1]

    avail = ort.get_available_providers()
    providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in avail]
    sess = ort.InferenceSession(args.model, providers=providers)
    rng = np.random.default_rng(0)
    if args.save_dir:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    for f in collect(args.inputs):
        bgr = cv2.imread(str(f))
        if bgr is None:
            print(f"{f}: unreadable")
            continue
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if args.cctv:
            img = cctv_degrade(img, rng)
        inp = letterbox(img, size)                                   # uint8 HWC RGB
        if args.save_dir:
            cv2.imwrite(str(Path(args.save_dir) / f.name), cv2.cvtColor(inp, cv2.COLOR_RGB2BGR))
        x = inp.transpose(2, 0, 1)[None].astype(np.float32)          # 0-255, NCHW
        prob = float(sess.run(["prob"], {"image": x})[0][0, 0])
        print(f"{f.name}: P(smoking)={prob:.3f} -> {'SMOKING' if prob >= thr else 'not smoking'} (thr {thr:.2f})")


if __name__ == "__main__":
    main()
