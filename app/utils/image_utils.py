"""
Image decode/resize/normalize helpers, plus segmentation-specific
mask <-> color conversion. Target size and normalization stats live
in config.py (per-project, per-model-backbone) -- this file just
applies them.
"""
import io

import numpy as np
from PIL import Image

from app.config import settings, CAMVID_PALETTE


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Decode -> resize to IMG_SIZE -> ImageNet-normalize -> CHW float32 array."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    h, w = settings.IMG_SIZE
    image = image.resize((w, h), Image.BILINEAR)

    array = np.asarray(image, dtype=np.float32) / 255.0
    mean = np.array(settings.IMAGENET_MEAN, dtype=np.float32)
    std = np.array(settings.IMAGENET_STD, dtype=np.float32)
    array = (array - mean) / std

    return array.transpose(2, 0, 1)  # HWC -> CHW


def decode_original_image(image_bytes: bytes) -> Image.Image:
    """Full-resolution RGB image, used for building the overlay (not resized)."""
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def mask_to_color(class_map: np.ndarray) -> Image.Image:
    """Map a (H, W) array of class ids to an (H, W, 3) RGB color image."""
    h, w = class_map.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, rgb in enumerate(CAMVID_PALETTE):
        color[class_map == class_id] = rgb
    return Image.fromarray(color)


def overlay_mask(base_image: Image.Image, class_map: np.ndarray, alpha: float = None) -> Image.Image:
    """Blend the colorized mask over the original image for visualization."""
    alpha = settings.OVERLAY_ALPHA if alpha is None else alpha
    color_mask = mask_to_color(class_map).resize(base_image.size, Image.NEAREST)
    return Image.blend(base_image, color_mask, alpha)


def encode_png_base64(image: Image.Image) -> str:
    import base64
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")
