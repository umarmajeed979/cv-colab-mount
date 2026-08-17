"""
Inference logic. This is the ONE place that turns raw bytes into a
prediction. routes.py calls this and nothing else -- keeps the API
layer swappable (FastAPI today, could be a CLI or Streamlit call
tomorrow) without touching model code.
"""
import time

import numpy as np
import torch

from app.config import settings, CAMVID_CLASSES
from app.core.model import get_model
from app.utils.image_utils import (
    preprocess_image,
    decode_original_image,
    mask_to_color,
    overlay_mask,
    encode_png_base64,
)
from app.api.schemas import PredictResponse, ClassCoverage


class Predictor:
    def __init__(self):
        self.model = get_model()
        self.labels = CAMVID_CLASSES

    def is_loaded(self) -> bool:
        return self.model is not None

    def _class_coverage(self, class_map: np.ndarray) -> list[ClassCoverage]:
        total = class_map.size
        ids, counts = np.unique(class_map, return_counts=True)
        coverage = [
            ClassCoverage(label=self.labels[i], pixel_percentage=round(100 * c / total, 2))
            for i, c in zip(ids, counts)
            if i < len(self.labels)
        ]
        return sorted(coverage, key=lambda x: x.pixel_percentage, reverse=True)

    def predict(self, image_bytes: bytes) -> PredictResponse:
        start = time.perf_counter()

        if self.model is None:
            raise RuntimeError("Model not loaded -- train and export weights first (see README.md).")

        original = decode_original_image(image_bytes)
        tensor = torch.from_numpy(preprocess_image(image_bytes)).unsqueeze(0).float().to(settings.DEVICE)

        with torch.no_grad():
            output = self.model(tensor)
            logits = output["out"] if isinstance(output, dict) else output
            class_map = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()

        overlay = overlay_mask(original, class_map)
        mask_only = mask_to_color(class_map).resize(original.size)

        elapsed_ms = (time.perf_counter() - start) * 1000
        return PredictResponse(
            overlay_image_base64=encode_png_base64(overlay),
            mask_image_base64=encode_png_base64(mask_only),
            class_coverage=self._class_coverage(class_map),
            inference_time_ms=round(elapsed_ms, 2),
            model_version=settings.VERSION,
        )
