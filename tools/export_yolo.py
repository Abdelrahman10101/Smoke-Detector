"""Export YOLO26n to ONNX with end2end NMS-free head."""
import argparse
from pathlib import Path
from ultralytics import YOLO


def export(out_dir: Path, imgsz: int = 640, opset: int = 17):
    out_dir.mkdir(parents=True, exist_ok=True)
    target_onnx = out_dir / "yolo26n.onnx"
    
    print(f"Loading YOLO26n and exporting to {target_onnx} (imgsz={imgsz}, opset={opset}, end2end=True)...")
    model = YOLO("yolo26n.pt")
    res = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=opset,
        simplify=True,
        end2end=True,
        dynamic=False,
        half=False
    )
    res_path = Path(res)
    if res_path.resolve() != target_onnx.resolve():
        import shutil
        shutil.copy2(res_path, target_onnx)
    print(f"Successfully exported YOLO26n ONNX to {target_onnx}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).parent.parent / "models")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()
    export(args.out_dir, args.imgsz, args.opset)
