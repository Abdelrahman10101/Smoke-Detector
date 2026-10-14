# Plan: MobileNetV3-Large Smoking vs Not-Smoking Classifier (CCTV-style preprocessing)

**Training:** PyTorch on Google Colab. **Deployment:** ONNX → TensorRT.

## 0. Decisions (confirmed)

| Topic | Decision |
|---|---|
| Backbone | **MobileNetV3-Large** (torchvision, ImageNet pretrained) |
| Framework | PyTorch (train) → ONNX → TensorRT (inference) |
| Test set | **Ignored** (`testing_data/` is not used anywhere) |
| Resizing | **Letterbox (pad), never crop** – so the cigarette is never cut out |
| Checkpoints | Saved to Colab local disk `/content/checkpoints/` (no Google Drive) |
| Optimizer / LR schedule | AdamW, ReduceLROnPlateau `min_lr=2.5e-4` |
| Early stopping | patience = 5 epochs |
| Epochs | up to 100 |

## 1. Dataset & Colab data access

| Item | Value |
|---|---|
| Source | `https://data.mendeley.com/public-api/zip/7b52hhzs3r/download/1` |
| `training_data/` | `smoking` 805 + `notsmoking` 805 = **1610** (balanced) |
| `validation_data/` | `smoking` 200 + `notsmoking` 200 = **400** (balanced) |
| Formats | `.jpg` (+ one `.jpeg`) |
| Sizes | 460×260 … 2400×1600, mixed landscape/portrait |

```python
!wget -q -O /content/smoking.zip "https://data.mendeley.com/public-api/zip/7b52hhzs3r/download/1"
!unzip -q /content/smoking.zip -d /content/raw
# recursively unzip any inner .zip, then auto-locate the folder containing
# training_data/ and validation_data/ (layout of the Mendeley zip is unverified)
```

- Data stays on Colab local disk (`/content`) for fast I/O.
- Runtime: **GPU (T4)**. PyTorch, torchvision, OpenCV, scikit-learn are preinstalled;
  only `onnx`, `onnxruntime`, (optional `onnxsim`) are `pip install`ed in the export cell.
- `testing_data/` is skipped when locating/auditing the dataset.

> [!WARNING]
> **`min_lr = 0.00025` is high.** It only matters if the starting LR is above it. With the usual
> fine-tuning LR (1e-4) ReduceLR would never fire. The plan starts at **1e-3** and halves:
> `1e-3 → 5e-4 → 2.5e-4 (floor)`; training then continues at 2.5e-4 until Early Stopping triggers.

## 2. Model

- `torchvision.models.mobilenet_v3_large(weights=IMAGENET1K_V2)` (V1 weights selectable).
- Replace the last layer: classifier `Linear(960→1280) → Hardswish → Dropout(0.2→0.3) → Linear(1280→1)`;
  output is a single logit, **`BCEWithLogitsLoss`**, `P(smoking)=sigmoid(logit)`.
- Input: **224×224** RGB, ImageNet mean/std normalisation (configurable to 256 for small cigarettes).
- MobileNetV3 notes: BatchNorm + Hardswish/Hardsigmoid are all ONNX/TensorRT-friendly; no custom ops.

## 3. Hyper-parameters

| Parameter | Value | Source |
|---|---|---|
| Optimizer | **AdamW**, `lr=1e-3`, `weight_decay=1e-2` | you / me |
| LR scheduler | **ReduceLROnPlateau**, `min_lr=2.5e-4`, `factor=0.5`, `patience=2`, monitor `val_loss` | you / me |
| Early stopping | **patience=5**, monitor `val_loss`, `min_delta=1e-4`, restore best weights | you / me |
| Max epochs | **100** | you |
| Batch size | 64 (T4 16 GB, AMP) | me |
| Input size | 224×224 | me |
| Loss | `BCEWithLogitsLoss` | me |
| Seed | 42 | me |
| Mixed precision | `torch.autocast` fp16 + `GradScaler` | me |

ReduceLR patience (2) < Early Stopping patience (5), so the LR gets to drop before training is stopped.
No weight decay on BatchNorm/bias parameters (AdamW param groups).

## 4. CCTV-style image preprocessing (core requirement)

Goal: make clean web photos look like surveillance frames, so the model matches the deployment domain.

### 4.1 Degradation pipeline (OpenCV + NumPy)

Applied in this order, each with its own probability and random magnitude:

1. **Resolution loss** – downscale to a CCTV-like size (long side 320–640 px, e.g. CIF 352×288, D1 640×480)
   with random interpolation, then upscale back.
2. **Lens/defocus blur** – Gaussian σ ∈ [0.3, 1.8] (p=0.6).
3. **Motion blur** – linear kernel 3–9 px, random angle (p=0.3).
4. **Sensor noise** – Gaussian σ ∈ [3, 20] (+ optional Poisson/salt-pepper) (p=0.7).
5. **Exposure & colour** – brightness ±35 %, contrast 0.6–1.2, gamma 0.6–1.6, colour cast, saturation 0.4–1.0 (p=0.8).
6. **Low-light / IR / night** – darken + noise, or grayscale with a faint tint (p≈0.2).
7. **Lens artefacts** – mild barrel distortion and vignette (p≈0.2).
8. **Compression artefacts** – JPEG re-encode, quality 15–50, 1–2 passes (H.264/MJPEG blocking) (p=0.85).
9. **Interlacing / scanlines** (p=0.1).
10. **Timestamp/camera-label overlay** in a corner (p=0.15).

> [!NOTE]
> The cigarette is small, so blur/downscale/noise magnitudes are **capped** to keep it
> at least partially visible; the preview grid is used to check this visually.

### 4.2 Letterbox resizing (replaces cropping)

Order of operations for every image:

```
load RGB → (train only) geometric aug → CCTV degradation
        → letterbox to 224×224 (aspect-preserving resize, pad with constant grey 114 / black)
        → ToTensor → Normalize(ImageNet mean/std)
```

- Degradation runs **before** padding, so the padding bars are not noised/blurred and look like
  clean borders (the same order is used at inference).
- **No `RandomResizedCrop` and no centre-crop.** Geometric augmentation is crop-free:
  horizontal flip, ±10° rotation and random scale 0.8–1.1 / translation ±8 % with padding fill
  (zoom-out only; never cuts content away).
- Pad colour is fixed (grey 114) and identical in train/val/inference.

### 4.3 Where each step is applied

| Split | Treatment |
|---|---|
| **Train** | On-the-fly: **80 % CCTV-degraded / 20 % clean**, crop-free geometric aug + flip, letterbox, normalise. Fresh random degradation every epoch. |
| **Validation** | **Fixed-seed** CCTV degradation per image (seed from filename hash) so metrics are comparable across epochs → stable ReduceLR / EarlyStopping signal. A **clean** validation score is also reported at the end. |

## 5. Data integrity checks

1. Verify every file opens (Pillow `verify()`), convert to RGB (grayscale/CMYK/odd formats).
2. Accept `.jpg .jpeg .png`.
3. **Leakage check** train ↔ validation with perceptual hash (pHash): report near-duplicates and
   (by default) drop them from training.
4. Print class balance and image-size histogram.
5. Save a **preview grid** (clean vs CCTV-degraded, 32 samples) for visual approval **before** training.

## 6. Training procedure

```
for epoch in 1..100:
    train one epoch (AMP, AdamW)
    validate -> val_loss, val_acc, val_f1, val_auc
    scheduler.step(val_loss)              # ReduceLROnPlateau (min_lr=2.5e-4)
    save last.pt every epoch              # /content/checkpoints/
    if val_loss improved: save best.pt
    early_stopping(val_loss)              # patience=5 -> stop, restore best
```

- Optional 2-epoch **backbone-frozen warm-up** to avoid damaging pretrained features at LR 1e-3
  (counts toward the 100 epochs; early stopping off during warm-up).
- Logging: per-epoch train/val loss, accuracy, F1, AUC, current LR → `history.csv`; plots
  `loss`, `accuracy`, `lr`.
- **Checkpoints (no Drive):** `last.pt` (model + optimizer + scheduler + early-stopping state + epoch)
  and `best.pt` in `/content/checkpoints/`. The notebook has a **resume cell** that works if the runtime
  is still alive (e.g. after interrupting a cell). 
- `num_workers=2` (Colab has ~2 CPU cores; the CCTV augmentation is the likely bottleneck).

> [!WARNING]
> Without Google Drive, checkpoints live on the Colab VM and are **lost if the runtime is
> disconnected/recycled**. The final cell zips `best.pt`, the ONNX model, plots and `history.csv` and
> triggers a browser download (`google.colab.files.download`). Keep the tab active; a smoke-test
> mode and a manual "download checkpoint" cell are provided for long runs.

## 7. Evaluation (validation only)

- Metrics on validation, both **CCTV-degraded** and **clean**: accuracy, precision, recall, F1, ROC-AUC,
  confusion matrix.
- **Threshold tuning** on validation: default 0.5, F1-maximising, and a threshold giving ≥ 95 % smoking recall.
  Chosen threshold is saved with the model metadata.
- Gallery of misclassified validation images.
- *(No test-set inference — `testing_data` is ignored.)*

## 8. Export for ONNX → TensorRT

1. Load `best.pt`, `eval()`, fp32.
2. **Bake normalisation into the model** (a wrapper `x/255 → (x-mean)/std → backbone → sigmoid`)
   so TensorRT receives raw 0–255 (or 0–1) letterboxed images; this removes a preprocessing
   mismatch risk. Letterbox/resize stays in application code (documented, with `letterbox()` helper).
3. `torch.onnx.export(..., opset_version=17, input_names=["image"], output_names=["prob"],
   dynamic_axes={"image": {0: "batch"}})`, fixed 224×224 spatial size.
4. Validate: `onnx.checker`, optional `onnxsim`, and **parity test** PyTorch vs ONNX Runtime on ≥ 50 validation
   images (max abs diff < 1e-4 on probabilities).
5. TensorRT notes (provided as commands/snippet, run by you on the target GPU, since Colab
   TensorRT versions vary):
   `trtexec --onnx=smoking_mobilenetv3.onnx --saveEngine=smoking.engine --fp16 --minShapes=image:1x3x224x224 --optShapes=image:8x3x224x224 --maxShapes=image:32x3x224x224`
   (build the engine on the **same GPU/TensorRT version** used for inference; INT8 optional with calibration
   from CCTV-degraded validation images).
6. `predict()` reference implementation (OpenCV letterbox + ONNX Runtime) included in the notebook.

## 9. Deliverable: one Colab notebook

`d:\Abdo\Task\PetroChoice\smoking_mobilenetv3_colab.ipynb`

| # | Cell | Content |
|---|---|---|
| 1 | Setup | GPU check, imports, seed, `CFG` (all hyper-parameters) |
| 2 | Data | download Mendeley zip → unzip → locate `training_data`/`validation_data` |
| 3 | Audit | corrupt-file check, counts, sizes, pHash leakage check |
| 4 | CCTV transforms | degradation functions, letterbox, train/val pipelines |
| 5 | **Preview** | clean vs CCTV grid (visual approval) |
| 6 | Datasets/loaders | labelled train/val datasets, fixed-seed val degradation |
| 7 | Model | MobileNetV3-Large, AdamW, ReduceLROnPlateau, EarlyStopping |
| 8 | Train | loop, local checkpoints, resume |
| 9 | Plots | loss / accuracy / LR curves |
| 10 | Evaluate | metrics, confusion matrix, threshold tuning, errors |
| 11 | Export | ONNX (+ normalisation baked in), parity test, TensorRT command |
| 12 | Package | zip artifacts + download |

A flat `.py` copy of the same code is also saved for reference.

## 10. Execution order & verification

| # | Step | Verification |
|---|---|---|
| 1 | Runtime → GPU, setup | `torch.cuda.is_available()` |
| 2 | Download, audit, leakage check | counts match §1 |
| 3 | CCTV + letterbox preview | **you approve the preview grid** |
| 4 | Smoke test (`CFG.smoke=True`: 1–2 epochs, 64 images) | loss falls, LR logged, no crash |
| 5 | Full training (≤ 100 epochs) | LR reaches 2.5e-4 and stays; stops 5 epochs after the best val loss |
| 6 | Evaluation + threshold | metrics and curves saved |
| 7 | ONNX export + parity test | max diff < 1e-4 |

Before handing over I will run the CCTV pipeline, letterbox, model, training loop, scheduler, early
stopping and ONNX export locally on a few synthetic/sample images to catch bugs (I cannot execute
the notebook on Colab itself). Expected Colab T4 runtime: ~20–40 s/epoch, typically 30–60 epochs.

## 11. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Cigarette lost by downscaling | Letterbox (no crop), capped degradation, optional 256 input, preview check |
| Over-degradation makes labels unlearnable | 20 % clean samples; capped blur/noise ranges |
| Random val degradation → noisy early stopping | Fixed per-image seeds on validation |
| Lost checkpoints on disconnect (no Drive) | `last.pt` every epoch, final zip/download, resume cell |
| `min_lr` ≥ LR (scheduler does nothing) | Start LR 1e-3, factor 0.5 |
| Overfit on 1610 images | Dropout, weight decay, augmentation, early stopping |
| Train/inference preprocessing mismatch | Normalisation baked into ONNX; shared `letterbox()` and pad value |
| TensorRT engine incompatibility | Build engine on the target machine from the ONNX file |
| Web-photo bias vs real CCTV | CCTV augmentation; recommend validating on a few real CCTV clips |

## 12. Remaining small choices (defaults used unless you object)

1. Input size **224** (switch to 256 if cigarettes are too small after letterbox).
2. Pad colour **grey 114**.
3. ImageNet weights **V2** for MobileNetV3-Large.
