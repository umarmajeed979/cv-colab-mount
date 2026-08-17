"""
Verifies the raw CamVid download is in place and builds the resized
processed/train|validation|test split (see app/data/preprocessor.py).

Expected raw layout (after downloading, see README.md "Dataset" --
no registration needed, plain GitHub files):

  data/raw/CamVid/{train,val,test}/*.png
  data/raw/CamVid/{trainannot,valannot,testannot}/*.png

Run:
  python scripts/prepare_data.py
"""
from pathlib import Path

from app.config import settings
from app.data.preprocessor import build_splits
from app.utils.logger import get_logger, setup_logging

setup_logging()  # scripts don't go through create_app(), so this must be called explicitly
logger = get_logger(__name__)


def main():
    camvid_root = Path(settings.RAW_DATA_DIR) / settings.CAMVID_SUBDIR
    if not camvid_root.exists():
        logger.error(
            f"{camvid_root} not found. Download CamVid with:\n"
            f"  git clone --depth 1 https://github.com/alexgkendall/SegNet-Tutorial {settings.RAW_DATA_DIR}/_segnet_tmp\n"
            f"  mv {settings.RAW_DATA_DIR}/_segnet_tmp/CamVid {camvid_root}\n"
            f"  rm -rf {settings.RAW_DATA_DIR}/_segnet_tmp\n"
            "(or download the zip from the repo's 'Code' button and extract the CamVid/ folder "
            f"to {camvid_root} manually). See README.md for details."
        )
        raise SystemExit(1)

    build_splits()
    logger.info("Done. Processed data written to " + settings.PROCESSED_DATA_DIR)


if __name__ == "__main__":
    main()
