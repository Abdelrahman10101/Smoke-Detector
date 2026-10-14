"""Export best.pt to ONNX (normalisation + sigmoid baked in) and verify against PyTorch.

    python export_onnx.py [--ckpt outputs/checkpoints/best.pt] [--out outputs/smoking_model.onnx]

ONNX input : "image"  float32 [N,3,H,W], RGB, values 0-255, letterboxed (pad=114)
ONNX output: "prob"   float32 [N,1]  = P(smoking)
"""
import argparse
import json

import numpy as np
import torch

from config import CKPT_DIR, MEAN_255, OUT_DIR, STD_255, VAL_SPLIT, resolve_data_root
from dataset import SmokingDataset, list_samples
from model import ExportWrapper, load_checkpoint_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(CKPT_DIR / "best.pt"))
    ap.add_argument("--out", default=None, help="Output ONNX filename (default: outputs/smoking_<arch>.onnx)")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--simplify", action="store_true", help="run onnxsim (pip install onnxsim)")
    ap.add_argument("--parity-images", type=int, default=50)
    args = ap.parse_args()

    import onnx
    import onnxruntime as ort

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model, cfg, ck = load_checkpoint_model(args.ckpt, "cpu")
    out_path = args.out or str(OUT_DIR / f"smoking_{cfg.arch}.onnx")
    wrapper = ExportWrapper(model).eval()
    size = cfg.img_size
    dummy = torch.randint(0, 256, (1, 3, size, size)).float()

    kwargs = dict(opset_version=args.opset, input_names=["image"], output_names=["prob"],
                  dynamic_axes={"image": {0: "batch"}, "prob": {0: "batch"}})
    try:  # legacy TorchScript exporter: most predictable for TensorRT (torch>=2.9 defaults to dynamo)
        torch.onnx.export(wrapper, dummy, out_path, dynamo=False, **kwargs)
    except TypeError:
        torch.onnx.export(wrapper, dummy, out_path, **kwargs)

    if args.simplify:
        import onnxsim
        m, ok = onnxsim.simplify(onnx.load(out_path))
        assert ok, "onnxsim failed"
        onnx.save(m, out_path)

    onnx.checker.check_model(onnx.load(out_path))
    print("ONNX saved & checked:", out_path)

    # ---- parity test ----
    try:
        paths, labels = list_samples(resolve_data_root(), VAL_SPLIT, use_exclusions=False)
        idx = np.random.default_rng(0).permutation(len(paths))[:args.parity_images]
        ds = SmokingDataset([paths[i] for i in idx], [labels[i] for i in idx], cfg, train=False, degrade=True)
        x = torch.from_numpy(ds.cache).permute(0, 3, 1, 2).float()
    except Exception:
        x = torch.randint(0, 256, (max(args.parity_images, 16), 3, size, size)).float()

    with torch.no_grad():
        ref = wrapper(x).numpy()
    sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    got = sess.run(["prob"], {"image": x.numpy()})[0]
    diff = float(np.abs(ref - got).max())
    print(f"PyTorch vs ONNX Runtime max |diff| over {len(x)} images: {diff:.2e}")
    assert diff < 1e-3, "ONNX output differs from PyTorch!"

    # dynamic batch check
    assert sess.run(["prob"], {"image": x.numpy()[:3]})[0].shape == (3, 1)

    thr = 0.5
    mpath = OUT_DIR / "metrics.json"
    if mpath.exists():
        thr = json.loads(mpath.read_text()).get("chosen_threshold", 0.5)
    meta = {"arch": cfg.arch, "input_name": "image", "output_name": "prob", "input_shape": ["N", 3, size, size],
            "input_format": "float32 NCHW RGB, values 0-255 (normalisation is inside the model)",
            "preprocess": f"letterbox to {size}x{size}, pad={cfg.pad_value}, aspect preserved, no crop",
            "mean_255": MEAN_255, "std_255": STD_255, "classes": ["notsmoking", "smoking"],
            "threshold": thr}
    meta_path = OUT_DIR / "model_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print("saved", meta_path)

    engine_name = f"smoking_{cfg.arch}_fp16.engine"
    cmd = (f"trtexec --onnx={out_path} --saveEngine={engine_name} --fp16 "
           f"--minShapes=image:1x3x{size}x{size} --optShapes=image:8x3x{size}x{size} "
           f"--maxShapes=image:32x3x{size}x{size}")
    (OUT_DIR / "build_trt_engine.txt").write_text(cmd + "\n")
    print("\nBuild the TensorRT engine ON THE TARGET GPU with:\n  " + cmd)


if __name__ == "__main__":
    main()
