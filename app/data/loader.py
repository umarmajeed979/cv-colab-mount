"""
Dataset loading utilities -- wraps a torch Dataset over the CamVid
11-class split (train/trainannot, val/valannot, test/testannot).
"""
from pathlib import Path
from typing import Optional, Callable

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from app.config import settings


def list_raw_files(extension: str = "*.png"):
    return list(Path(settings.RAW_DATA_DIR).rglob(extension))


class ProcessedSegmentationDataset(Dataset):
    """
    Reads the resized image/mask pairs written by app/data/preprocessor.py
    into PROCESSED_DATA_DIR/<split>/{images,masks}/*.png -- this is what
    training and evaluation actually load from (CamVidSegmentationDataset
    above reads straight from the raw download instead).

    Lives here (in app/, not scripts/) specifically so it's importable
    from anywhere: `app` is installed via `pip install -e .`, so
    `from app.data.loader import ProcessedSegmentationDataset` works
    regardless of your current directory or how a script was invoked.
    `scripts/` is NOT an installed package, so anything defined only
    inside a scripts/*.py file is only reliably importable by that exact
    script itself.
    """

    def __init__(self, split_dir: Path):
        self.images_dir = split_dir / "images"
        self.masks_dir = split_dir / "masks"
        self.stems = sorted(p.stem for p in self.images_dir.glob("*.png"))
        self.mean = np.array(settings.IMAGENET_MEAN, dtype=np.float32)
        self.std = np.array(settings.IMAGENET_STD, dtype=np.float32)

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx):
        import torch  # local import: keeps torch out of this module's top-level

        stem = self.stems[idx]
        image = Image.open(self.images_dir / f"{stem}.png").convert("RGB")
        mask = Image.open(self.masks_dir / f"{stem}.png")

        image = (np.asarray(image, dtype=np.float32) / 255.0 - self.mean) / self.std
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        mask = torch.from_numpy(np.array(mask, dtype=np.int64))
        return image, mask


class CamVidSegmentationDataset(Dataset):
    """
    Pairs each image in `<raw>/CamVid/<split>/*.png` with its label mask
    in `<raw>/CamVid/<split>annot/*.png` (same filename in both).

    CamVid annot masks are single-channel PNGs where pixel value ==
    class id (0-10), with 11 marking "Unlabelled/void" -- no color
    decoding needed, values are already class indices.
    """

    def __init__(self, split: str = "train", transform: Optional[Callable] = None):
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be train/val/test, got {split!r}")

        self.split = split
        self.transform = transform

        camvid_root = Path(settings.RAW_DATA_DIR) / settings.CAMVID_SUBDIR
        images_dir = camvid_root / split
        labels_dir = camvid_root / f"{split}annot"

        if not images_dir.exists():
            raise FileNotFoundError(
                f"{images_dir} not found -- see README.md 'Dataset' section for the "
                "CamVid download command (plain GitHub files, no registration)."
            )

        self.samples = []
        for img_path in sorted(images_dir.glob("*.png")):
            label_path = labels_dir / img_path.name
            if label_path.exists():
                self.samples.append((img_path, label_path))

        if not self.samples:
            raise RuntimeError(f"No matched image/label pairs found for split={split!r}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label_path = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(label_path)  # single-channel, values = class ids / 11=ignore

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        return image, np.array(mask, dtype=np.int64)
