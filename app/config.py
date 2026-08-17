"""
Single source of truth for config. No hardcoded values anywhere else
in the codebase -- everything that can change between projects or
environments (model path, thresholds, dataset dir) lives here and is
read from .env.
"""
from pydantic_settings import BaseSettings
from typing import List, Tuple


# CamVid label set (SegNet 11-class convention): 11 trainable classes
# (ids 0-10) + id 11 = "Unlabelled"/void, ignored during loss + metrics.
# Names, ids, and RGB values match Scripts/test_segmentation_camvid.py in
# the source repo exactly (see README.md "Dataset").
CAMVID_CLASSES: List[str] = [
    "sky", "building", "pole", "road", "pavement", "tree",
    "sign_symbol", "fence", "car", "pedestrian", "bicyclist",
]

CAMVID_PALETTE: List[Tuple[int, int, int]] = [
    (128, 128, 128), (128, 0, 0), (192, 192, 128), (128, 64, 128), (60, 40, 222),
    (128, 128, 0), (192, 128, 128), (64, 64, 128), (64, 0, 128), (64, 64, 0), (0, 128, 192),
]

IGNORE_INDEX = 11  # "Unlabelled" class id in the source masks


class Settings(BaseSettings):
    # --- identity (change per project) ---
    PROJECT_NAME: str = "semantic-segmentation-autonomous-driving"
    VERSION: str = "0.1.0"
    DESCRIPTION: str = "Road scene semantic segmentation (CamVid, 11-class SegNet taxonomy) for autonomous driving."

    # --- paths ---
    MODEL_PATH: str = "data/models/model_final.pt"
    TORCHSCRIPT_MODEL_PATH: str = "data/models/model_final.torchscript.pt"
    CLASS_LABELS_PATH: str = "data/models/class_labels.json"
    RAW_DATA_DIR: str = "data/raw"
    PROCESSED_DATA_DIR: str = "data/processed"

    # --- dataset (CamVid, SegNet 11-class split) ---
    # No registration needed -- plain files from github.com/alexgkendall/SegNet-Tutorial.
    # See README.md "Dataset" for the exact download command. Expected layout once
    # extracted under RAW_DATA_DIR: CamVid/{train,trainannot,val,valannot,test,testannot}/*.png
    CAMVID_SUBDIR: str = "CamVid"
    NUM_CLASSES: int = 11
    IGNORE_INDEX: int = IGNORE_INDEX

    # --- model / training ---
    ARCHITECTURE: str = "deeplabv3_resnet50"  # or "unet"
    IMG_SIZE: Tuple[int, int] = (360, 480)  # (H, W) -- CamVid's native resolution, no downsizing needed
    IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    # --- inference ---
    CONFIDENCE_THRESHOLD: float = 0.5  # not used for argmax segmentation, kept for API parity
    DEVICE: str = "cpu"  # "cuda" if available, set at runtime in predictor.py
    OVERLAY_ALPHA: float = 0.5  # blend weight for mask-over-image visualization

    # --- api ---
    CORS_ORIGINS: List[str] = ["*"]
    MAX_UPLOAD_SIZE_MB: int = 10

    # --- logging ---
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"


settings = Settings()
