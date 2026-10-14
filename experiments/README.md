# Smoking vs Not-Smoking Classifier (CCTV-style training)

PyTorch training on Google Colab → ONNX → TensorRT.
Supports **EfficientNet-B0**, **EfficientNet-V2-S**, and **MobileNetV3-Large**.
AdamW · ReduceLROnPlateau (`min_lr=2.5e-4`) · EarlyStopping (patience 5) · up to 100 epochs.

## Colab usage

Upload this folder to Colab (e.g. `/content/experiments`), set **Runtime → GPU**, then:

```bash
%cd /content/experiments
!apt-get install -y unrar                 # required for Mendeley's inner .rar archive
!pip install -q onnx onnxruntime          # everything else is preinstalled on Colab
!python download_data.py                  # Mendeley zip -> data/, auto-locates the dataset root
!python audit_data.py                     # corrupt files, sizes, train/val leakage (writes outputs/exclude.txt)
!python preview_cctv.py                   # CHECK outputs/preview_cctv_*.png before training

# Train EfficientNet-B0 (default) or MobileNetV3:
!python train.py --arch efficientnet_b0   # (or: --arch efficientnet_v2_s, --arch mobilenet_v3_large)

!python evaluate.py                       # metrics (CCTV + clean), threshold tuning, error gallery
!python export_onnx.py                    # ONNX + parity test + model_meta.json + trtexec command
!zip -qr results.zip outputs -x "outputs/checkpoints/last.pt"
from google.colab import files; files.download("results.zip")
```

Resume an interrupted run: `!python train.py --resume`.

## Architecture Options

Pass any architecture via `--arch`:
- `--arch efficientnet_b0` (Efficient, high accuracy, standard 224px)
- `--arch efficientnet_v2_s` (Faster convergence, strong modern features)
- `--arch mobilenet_v3_large` (Ultra lightweight for edge devices)
