"""
Trains the segmentation model on the processed CamVid split. Meant
to run on Colab (GPU) -- see README.md for the Colab setup snippet.

Persists two things every epoch (not just at the end -- a Colab
disconnect shouldn't cost you the whole run's history):
  logs/training_history.csv   -- one row per epoch, re-loadable/plottable anytime
  logs/training_curves.png    -- loss + mIoU/Dice curves, regenerated each epoch

Run:
  python scripts/train_model.py --epochs 30 --batch-size 8 --lr 1e-4
"""
import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from app.config import settings
from app.core.model import build_architecture
from app.core.evaluator import evaluate
from app.data.loader import ProcessedSegmentationDataset
from app.utils.logger import get_logger, setup_logging

setup_logging()  # scripts don't go through create_app(), so this must be called explicitly
logger = get_logger(__name__)

HISTORY_CSV = Path("logs/training_history.csv")
CURVES_PNG = Path("logs/training_curves.png")
HISTORY_FIELDS = ["epoch", "timestamp", "train_loss", "val_mean_iou", "val_mean_dice", "lr"]


def _append_history_row(row: dict) -> None:
    """Open-append-close every call (not once at the end) so a row is safely
    on disk the moment an epoch finishes, even if the process dies right after."""
    HISTORY_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not HISTORY_CSV.exists()
    with open(HISTORY_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _update_curves_plot() -> None:
    """Re-read the CSV and regenerate the plot from scratch -- cheap at this
    dataset's scale, and means the PNG on disk is always current, not just
    "as of the last successful full run"."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # no display needed, just save to file
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed -- skipping training_curves.png (history CSV is still written)")
        return

    epochs, losses, mious, mdices = [], [], [], []
    with open(HISTORY_CSV) as f:
        for row in csv.DictReader(f):
            epochs.append(int(row["epoch"]))
            losses.append(float(row["train_loss"]))
            mious.append(float(row["val_mean_iou"]))
            mdices.append(float(row["val_mean_dice"]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(epochs, losses, marker="o")
    ax1.set_title("Train loss")
    ax1.set_xlabel("epoch")
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, mious, marker="o", label="val mIoU")
    ax2.plot(epochs, mdices, marker="o", label="val mDice")
    ax2.set_title("Validation metrics")
    ax2.set_xlabel("epoch")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(CURVES_PNG, dpi=120)
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    # Default 0, not 2: Colab's DataLoader workers commonly hang indefinitely
    # with num_workers>0 (limited /dev/shm + fork-after-CUDA-init issues) --
    # this dataset is small enough that num_workers=0 is plenty fast. Only
    # raise this if you've confirmed >0 doesn't hang in your specific runtime.
    p.add_argument("--num-workers", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
    logger.info(f"Training on device={device} ({device_name}), architecture={settings.ARCHITECTURE}, num_workers={args.num_workers}")

    processed = Path(settings.PROCESSED_DATA_DIR)
    train_ds = ProcessedSegmentationDataset(processed / "train")
    val_ds = ProcessedSegmentationDataset(processed / "validation")
    logger.info(f"train={len(train_ds)} images, val={len(val_ds)} images")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_architecture(pretrained_backbone=True).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=settings.IGNORE_INDEX)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_miou = 0.0
    models_dir = Path(settings.MODEL_PATH).parent
    models_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        # tqdm here is the key change: previously nothing printed until a
        # FULL epoch (train + eval) finished, so a hang looked identical to
        # "just slow" -- now every batch updates a visible progress bar.
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", unit="batch")
        for images, masks in pbar:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            output = model(images)
            logits = output["out"] if isinstance(output, dict) else output
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        avg_loss = running_loss / len(train_ds)

        settings.DEVICE = device  # so evaluate()/predictor.py pick up the active device
        metrics = evaluate(model, val_loader)
        logger.info(
            f"epoch {epoch}/{args.epochs} loss={avg_loss:.4f} "
            f"val_mIoU={metrics['mean_iou']:.4f} val_mDice={metrics['mean_dice']:.4f}"
        )

        _append_history_row({
            "epoch": epoch,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "train_loss": round(avg_loss, 6),
            "val_mean_iou": round(metrics["mean_iou"], 6),
            "val_mean_dice": round(metrics["mean_dice"], 6),
            "lr": scheduler.get_last_lr()[0],
        })
        _update_curves_plot()

        if metrics["mean_iou"] > best_miou:
            best_miou = metrics["mean_iou"]
            torch.save(model.state_dict(), settings.MODEL_PATH)
            logger.info(f"New best mIoU={best_miou:.4f} -> saved {settings.MODEL_PATH}")

    logger.info(f"Training complete. Best val mIoU={best_miou:.4f}")


if __name__ == "__main__":
    main()
