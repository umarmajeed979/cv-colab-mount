"""
Identical in every project. Import get_logger(__name__) anywhere
instead of print() -- gives consistent formatting + log level
control via .env across the whole portfolio.
"""
import logging
import sys
from pathlib import Path
from app.config import settings


def setup_logging():
    # FileHandler does NOT create missing parent directories -- and git
    # doesn't track empty folders, so a fresh clone (e.g. on Colab) can
    # be missing logs/ entirely even though it exists locally. Always
    # ensure it's there before opening any file inside it.
    Path("logs").mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")

    error_handler = logging.FileHandler("logs/error.log")
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)

    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("logs/app.log"),
            error_handler,
        ],
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
