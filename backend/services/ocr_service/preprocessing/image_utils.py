"""
preprocessing/image_utils.py

Basic image I/O and normalization utilities:
- load_image
- normalize_image
- pad_and_crop (safe cropping with padding, clipped to image bounds)
"""

import os
import cv2
import numpy as np

from backend.services.ocr_service import config


def load_image_from_bytes(image_bytes):
    """
    Decodes an in-memory byte buffer (e.g. from Flask request.files) into
    a BGR numpy image array. Raises ValueError if decoding fails.
    """
    if not image_bytes:
        raise ValueError("Cannot load image from empty byte buffer.")
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("OpenCV failed to decode image bytes. Supported formats: JPG, PNG, WEBP.")
    return image


def load_image(path_or_bytes):
    """
    Load an image from disk path OR from raw in-memory bytes / numpy array.
    Raises FileNotFoundError or ValueError on invalid input.
    """
    if isinstance(path_or_bytes, (bytes, bytearray)):
        return load_image_from_bytes(path_or_bytes)

    if isinstance(path_or_bytes, np.ndarray):
        return path_or_bytes.copy()

    path = str(path_or_bytes)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Could not load image: {path} (file does not exist)")

    ext = os.path.splitext(path)[1].lower()
    if ext not in config.VALID_EXTENSIONS:
        raise ValueError(
            f"Unsupported file extension '{ext}'. Supported: {sorted(config.VALID_EXTENSIONS)}"
        )

    image = cv2.imread(path, cv2.IMREAD_COLOR)

    if image is None:
        raise FileNotFoundError(
            f"Could not load image: {path} (file exists but OpenCV failed to decode it)"
        )

    return image


def normalize_image(image, min_height=None, max_dim=None):
    """
    Normalize image size for reliable OCR:
      1. Check dimensions.
      2. Upscale small images (height < min_height) using INTER_CUBIC.
      3. Downscale very large images (max dimension > max_dim).
      4. Preserve aspect ratio throughout.

    Returns the normalized image (a new array; input is not mutated).
    """
    if image is None or image.size == 0:
        raise ValueError("normalize_image received an empty image")

    min_height = min_height or config.MIN_USEFUL_OCR_HEIGHT
    max_dim = max_dim or config.MAX_IMAGE_DIMENSION

    h, w = image.shape[:2]
    out = image

    # Step 1: upscale if too small for reliable OCR
    if h < min_height:
        scale = min_height / float(h)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        out = cv2.resize(out, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        h, w = new_h, new_w

    # Step 2: downscale if the longer side exceeds max_dim
    longer_side = max(h, w)
    if longer_side > max_dim:
        scale = max_dim / float(longer_side)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        out = cv2.resize(out, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return out


def clip_bbox(bbox, image_shape):
    """
    Clip an [x1, y1, x2, y2] bbox to valid image bounds.
    """
    h, w = image_shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(int(round(x1)), w - 1))
    y1 = max(0, min(int(round(y1)), h - 1))
    x2 = max(0, min(int(round(x2)), w))
    y2 = max(0, min(int(round(y2)), h))
    if x2 <= x1:
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        y2 = min(h, y1 + 1)
    return [x1, y1, x2, y2]


def pad_bbox(bbox, image_shape, padding_ratio=None, padding_ratio_x=None, padding_ratio_y=None):
    """
    Expand a bbox by a small padding ratio (relative to image size), then
    clip to image bounds. `padding_ratio` sets both axes;
    `padding_ratio_x`/`padding_ratio_y` override it per-axis. Defaults to
    config.REGION_PADDING_RATIO_X/Y (~1-2% of image size) - this is meant
    to be the ONE place a tight, already-semantically-bounded region gets
    a small visual margin before the actual crop, not a substitute for
    tight region detection.
    """
    padding_ratio = (
        padding_ratio if padding_ratio is not None else config.REGION_PADDING_RATIO
    )
    padding_ratio_x = padding_ratio_x if padding_ratio_x is not None else padding_ratio
    padding_ratio_y = padding_ratio_y if padding_ratio_y is not None else padding_ratio
    h, w = image_shape[:2]
    pad_x = int(round(w * padding_ratio_x))
    pad_y = int(round(h * padding_ratio_y))

    x1, y1, x2, y2 = bbox
    x1 -= pad_x
    y1 -= pad_y
    x2 += pad_x
    y2 += pad_y

    return clip_bbox([x1, y1, x2, y2], image_shape)


def safe_crop(image, bbox):
    """
    Crop image to bbox = [x1, y1, x2, y2], clipping to valid bounds first.
    Returns None if the resulting crop would be empty.
    """
    x1, y1, x2, y2 = clip_bbox(bbox, image.shape)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return image[y1:y2, x1:x2].copy()


def save_image(image, path):
    """
    Save image to disk, creating parent directories as needed.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, image)
    return path


def check_image_quality(image):
    """
    Lightweight quality check used before heavy processing. Returns a dict
    with blur score (variance of Laplacian), brightness, contrast, glare,
    shadow, and simple flags.
    This does not block the pipeline - it only informs downstream decisions
    and is written to result.json for debugging.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    
    # Glare: fraction of pixels with value > 250
    glare_score = float(np.mean(gray > 250))
    
    # Shadow: standard deviation of local 4x4 block means
    h, w = gray.shape
    bh, bw = max(2, h // 4), max(2, w // 4)
    block_means = []
    for r in range(4):
        for c in range(4):
            block = gray[r*bh:(r+1)*bh, c*bw:(c+1)*bw]
            if block.size > 0:
                block_means.append(np.mean(block))
    shadow_score = float(np.std(block_means)) if block_means else 0.0

    is_blurry = laplacian_var < 60.0
    is_too_dark = brightness < 40.0
    is_too_bright = brightness > 235.0

    return {
        "laplacian_variance": float(laplacian_var),
        "brightness_mean": brightness,
        "contrast_std": contrast,
        "glare_ratio": glare_score,
        "shadow_std": shadow_score,
        "is_blurry": bool(is_blurry),
        "is_too_dark": bool(is_too_dark),
        "is_too_bright": bool(is_too_bright),
    }


def assess_image_quality_gate(image, quality_dict=None, ocr_items=None):
    """
    Evaluates whether an image is usable for label reading.
    Conservative gate: avoids rejecting normal or imperfect packaging photos.
    Only rejects genuinely unusable images:
    - severe blur (laplacian_var < 35.0 AND contrast_std < 20.0)
    - extremely low resolution (w < 80 or h < 80 or w*h < 10000)
    - almost completely blank/uniform image (contrast_std < 4.0 or laplacian_var < 4.0)

    Returns:
        (is_usable: bool, status: str, message: str, details: dict)
    """
    if image is None or image.size == 0:
        return (
            False,
            "unusable_image",
            "Image quality is too low to reliably read the label. Please upload a clearer, sharper photo of the product label.",
            {"reason": "empty_image"},
        )

    h, w = image.shape[:2]
    total_pixels = h * w

    if quality_dict is None:
        quality_dict = check_image_quality(image)

    lap_var = quality_dict.get("laplacian_variance", 0.0)
    contrast = quality_dict.get("contrast_std", 0.0)
    brightness = quality_dict.get("brightness_mean", 0.0)

    # 1. Extremely low resolution
    if w < 80 or h < 80 or total_pixels < 10000:
        return (
            False,
            "unusable_image",
            "Image quality is too low to reliably read the label. Please upload a clearer, sharper photo of the product label.",
            {"reason": "extremely_low_resolution", "dimensions": [w, h], "pixels": total_pixels},
        )

    # 2. Blank / uniform image (nearly 0 contrast or 0 variance)
    if contrast < 4.0 or (lap_var < 4.0 and (brightness < 20.0 or brightness > 235.0)):
        return (
            False,
            "unusable_image",
            "Image quality is too low to reliably read the label. Please upload a clearer, sharper photo of the product label.",
            {"reason": "blank_or_uniform_image", "contrast_std": contrast, "laplacian_variance": lap_var},
        )

    # 3. Severe blur combined with low contrast
    if lap_var < 35.0 and contrast < 20.0:
        return (
            False,
            "unusable_image",
            "Image quality is too low to reliably read the label. Please upload a clearer, sharper photo of the product label.",
            {"reason": "severe_blur", "laplacian_variance": lap_var, "contrast_std": contrast},
        )

    return (
        True,
        "usable",
        "",
        {"laplacian_variance": lap_var, "contrast_std": contrast, "dimensions": [w, h]},
    )
