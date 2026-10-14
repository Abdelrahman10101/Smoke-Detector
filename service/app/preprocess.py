"""Preprocessing utilities: letterbox resize, coordinate transformation, and crop extraction."""
from typing import Tuple, List
import cv2
import numpy as np

PAD_VALUE = 114


def letterbox(
    img: np.ndarray,
    target_size: int = 640,
    pad_val: int = PAD_VALUE
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """Aspect-preserving resize and center padding.
    
    Returns:
        padded_img: uint8 image of shape (target_size, target_size, 3)
        scale: scale factor applied to the original image
        (pad_x, pad_y): left and top padding offsets in pixels
    """
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (nw, nh), interpolation=interp)
    
    out = np.full((target_size, target_size, 3), pad_val, dtype=np.uint8)
    top, left = (target_size - nh) // 2, (target_size - nw) // 2
    out[top:top + nh, left:left + nw] = resized
    return out, scale, (left, top)


def unletterbox_box(
    box: List[float] | np.ndarray,
    scale: float,
    pad: Tuple[int, int],
    orig_shape: Tuple[int, int]
) -> List[int]:
    """Map a bounding box [x1, y1, x2, y2] from letterboxed space back to original image space."""
    left, top = pad
    orig_h, orig_w = orig_shape
    x1, y1, x2, y2 = box[:4]
    
    x1_orig = (x1 - left) / scale
    y1_orig = (y1 - top) / scale
    x2_orig = (x2 - left) / scale
    y2_orig = (y2 - top) / scale
    
    # Clip to image boundaries
    x1_c = int(max(0, min(orig_w - 1, round(x1_orig))))
    y1_c = int(max(0, min(orig_h - 1, round(y1_orig))))
    x2_c = int(max(x1_c + 1, min(orig_w, round(x2_orig))))
    y2_c = int(max(y1_c + 1, min(orig_h, round(y2_orig))))
    
    return [x1_c, y1_c, x2_c, y2_c]


def extract_person_crop(
    img: np.ndarray,
    bbox: List[int],
    mode: str = "person",
    margin: float = 0.10
) -> np.ndarray:
    """Extract person crop from RGB image based on the selected mode.
    
    Args:
        img: RGB image (H, W, 3)
        bbox: [x1, y1, x2, y2] in original image coordinates
        mode: "person" | "upper_body" | "full_frame"
        margin: fractional margin to expand the bounding box
    """
    if mode == "full_frame":
        return img

    h_img, w_img = img.shape[:2]
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1

    if mode == "upper_body":
        # Upper body: top 65% of the height (head, chest, hands)
        y2_eff = y1 + h * 0.65
        mx = w * margin
        my = h * margin * 0.5
        x1_m = max(0, int(round(x1 - mx)))
        y1_m = max(0, int(round(y1 - my)))
        x2_m = min(w_img, int(round(x2 + mx)))
        y2_m = min(h_img, int(round(y2_eff + my)))
    else:
        # Default "person" mode: full person box with margin
        mx = w * margin
        my = h * margin
        x1_m = max(0, int(round(x1 - mx)))
        y1_m = max(0, int(round(y1 - my)))
        x2_m = min(w_img, int(round(x2 + mx)))
        y2_m = min(h_img, int(round(y2 + my)))

    # Fallback if coordinates collapse
    if x2_m <= x1_m or y2_m <= y1_m:
        return img[max(0, y1):min(h_img, y2), max(0, x1):min(w_img, x2)]

    return img[y1_m:y2_m, x1_m:x2_m]


def prepare_detector_input(rgb_img: np.ndarray, imgsz: int = 640) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """Letterbox and convert to float32 NCHW (0.0 to 1.0) for YOLO26n."""
    lb_img, scale, pad = letterbox(rgb_img, target_size=imgsz)
    tensor = lb_img.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return tensor, scale, pad


def prepare_classifier_input(rgb_crop: np.ndarray, imgsz: int = 224) -> np.ndarray:
    """Letterbox crop and convert to float32 CHW (0.0 to 255.0) for EfficientNet-B0.
    
    CRITICAL: The EfficientNet-B0 ONNX model has normalization (Sub(mean)/Div(std))
    baked in with ImageNet mean (123.675, 116.28, 103.53) and std (58.395, 57.12, 57.375).
    Input must remain in 0.0 - 255.0 range!
    """
    lb_crop, _, _ = letterbox(rgb_crop, target_size=imgsz, pad_val=PAD_VALUE)
    # Return (3, H, W) float32 in range 0-255
    return lb_crop.transpose(2, 0, 1).astype(np.float32)
