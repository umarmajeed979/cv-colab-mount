"""
Pydantic request/response contracts. Keep these separate from the
model's internal data structures (in core/) so the API shape doesn't
change just because the model architecture changes.
"""
from pydantic import BaseModel
from typing import List, Dict


class ClassCoverage(BaseModel):
    label: str
    pixel_percentage: float


class PredictResponse(BaseModel):
    overlay_image_base64: str  # original image with colorized mask blended in, PNG
    mask_image_base64: str  # raw colorized segmentation mask only, PNG
    class_coverage: List[ClassCoverage]  # % of image pixels per detected class, sorted desc
    inference_time_ms: float
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
