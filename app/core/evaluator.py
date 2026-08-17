"""
Offline evaluation: per-class IoU, mean IoU, and Dice, computed from a
confusion matrix accumulated over the whole loader. Run via
scripts/evaluate_model.py, not part of the live API.
"""
import numpy as np
import torch
from tqdm import tqdm

from app.config import settings, CAMVID_CLASSES
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _confusion_matrix(pred: np.ndarray, target: np.ndarray, num_classes: int) -> np.ndarray:
    mask = target != settings.IGNORE_INDEX
    pred, target = pred[mask], target[mask]
    idx = num_classes * target.astype(np.int64) + pred.astype(np.int64)
    return np.bincount(idx, minlength=num_classes ** 2).reshape(num_classes, num_classes)


def evaluate(model, data_loader) -> dict:
    """Return per-class IoU/Dice + mean IoU. Log + return, don't print."""
    device = settings.DEVICE
    model.eval()
    num_classes = settings.NUM_CLASSES
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="evaluating", unit="batch", leave=False):
            images = images.to(device)
            output = model(images)
            logits = output["out"] if isinstance(output, dict) else output
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            confusion += _confusion_matrix(preds, targets.numpy(), num_classes)

    tp = np.diag(confusion)
    fp = confusion.sum(axis=0) - tp
    fn = confusion.sum(axis=1) - tp

    iou = tp / np.maximum(tp + fp + fn, 1)
    dice = 2 * tp / np.maximum(2 * tp + fp + fn, 1)

    metrics = {
        "mean_iou": float(np.mean(iou)),
        "mean_dice": float(np.mean(dice)),
        "per_class_iou": {CAMVID_CLASSES[i]: round(float(iou[i]), 4) for i in range(num_classes)},
        "per_class_dice": {CAMVID_CLASSES[i]: round(float(dice[i]), 4) for i in range(num_classes)},
    }
    logger.info(f"Evaluation results: mean_iou={metrics['mean_iou']:.4f}, mean_dice={metrics['mean_dice']:.4f}")
    return metrics
