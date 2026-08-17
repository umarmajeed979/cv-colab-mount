"""
Turns raw CamVid files into the processed/train|validation|test split
used by training. CamVid already ships pre-split (train/val/test)
directories with full labels for all three (unlike most driving
segmentation sets, it has real test-set annotations too), so this is
just: verify pairs, resize image + mask together (nearest-neighbor for
the mask so label ids never blur -- a no-op at the default IMG_SIZE
since it matches CamVid's native 480x360, but keeps this safe if
IMG_SIZE is changed later), and write both out as PNG pairs ready for
fast loading. Run once via scripts/prepare_data.py, not at request time.
"""
from pathlib import Path

from PIL import Image

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# CamVid's own split names map directly onto ours; "validation" (config's val
# split name) is CamVid's "val".
_CAMVID_TO_OUR_SPLIT = {"train": "train", "val": "validation", "test": "test"}


def _resize_pair(image: Image.Image, mask: Image.Image, size_hw):
    h, w = size_hw
    image = image.resize((w, h), Image.BILINEAR)
    mask = mask.resize((w, h), Image.NEAREST)  # NEAREST: never interpolate label ids
    return image, mask


def build_splits(raw_dir: str = None, processed_dir: str = None, val_ratio: float = 0.15, test_ratio: float = 0.15):
    """
    Signature kept generic (matches the template contract) but CamVid
    doesn't need ratio-based splitting -- val_ratio/test_ratio are
    accepted and ignored so this still plugs into the shared script
    interface without special-casing prepare_data.py.
    """
    raw_dir = Path(raw_dir or settings.RAW_DATA_DIR)
    processed_dir = Path(processed_dir or settings.PROCESSED_DATA_DIR)
    camvid_root = raw_dir / settings.CAMVID_SUBDIR

    for camvid_split, our_split in _CAMVID_TO_OUR_SPLIT.items():
        images_dir = camvid_root / camvid_split
        labels_dir = camvid_root / f"{camvid_split}annot"

        if not images_dir.exists():
            logger.warning(f"Skipping split={camvid_split!r}: {images_dir} not found")
            continue

        out_images = processed_dir / our_split / "images"
        out_masks = processed_dir / our_split / "masks"
        out_images.mkdir(parents=True, exist_ok=True)
        out_masks.mkdir(parents=True, exist_ok=True)

        count = 0
        for img_path in sorted(images_dir.glob("*.png")):
            label_path = labels_dir / img_path.name
            if not label_path.exists():
                continue

            image = Image.open(img_path).convert("RGB")
            mask = Image.open(label_path)
            image, mask = _resize_pair(image, mask, settings.IMG_SIZE)

            image.save(out_images / img_path.name)
            mask.save(out_masks / img_path.name)
            count += 1

        logger.info(f"{camvid_split} -> {our_split}: wrote {count} image/mask pairs to {processed_dir / our_split}")
