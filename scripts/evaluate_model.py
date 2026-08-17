"""
Runs offline evaluation (mean IoU / Dice, per-class breakdown) against
the validation split using the currently saved weights at
settings.MODEL_PATH. Prints the result AND appends it to
logs/eval_history.jsonl, so repeated runs (e.g. after retraining) build
up a comparable history instead of each result only living in your
terminal scrollback.

Run:
  python scripts/evaluate_model.py
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from app.config import settings
from app.core.model import build_architecture
from app.core.evaluator import evaluate
from app.data.loader import ProcessedSegmentationDataset
from app.utils.logger import get_logger, setup_logging

setup_logging()  # scripts don't go through create_app(), so this must be called explicitly
logger = get_logger(__name__)

EVAL_HISTORY_PATH = Path("logs/eval_history.jsonl")


def _append_eval_record(metrics: dict) -> None:
    """One JSON object per line (JSONL) -- append-only, easy to re-read with
    a one-liner (`[json.loads(l) for l in open(path)]`), and safe to append
    to concurrently without needing to parse+rewrite the whole file each time
    (unlike a single growing JSON array or CSV with a changing column set)."""
    EVAL_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_path": settings.MODEL_PATH,
        "architecture": settings.ARCHITECTURE,
        **metrics,
    }
    with open(EVAL_HISTORY_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    settings.DEVICE = device

    val_ds = ProcessedSegmentationDataset(Path(settings.PROCESSED_DATA_DIR) / "validation")
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)

    model = build_architecture(pretrained_backbone=False).to(device)
    model.load_state_dict(torch.load(settings.MODEL_PATH, map_location=device))

    metrics = evaluate(model, val_loader)
    print(json.dumps(metrics, indent=2))

    _append_eval_record(metrics)
    logger.info(f"Appended this run to {EVAL_HISTORY_PATH}")


if __name__ == "__main__":
    main()
