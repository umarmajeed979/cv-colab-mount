"""
Converts the trained state_dict at settings.MODEL_PATH into a
TorchScript module for fast, dependency-light deployment inference
(matches the project's stated TorchScript tech-stack choice, no ONNX
conversion here).

Run:
  python scripts/export_model.py
"""
import torch

from app.config import settings
from app.core.model import build_architecture
from app.utils.logger import get_logger, setup_logging

setup_logging()  # scripts don't go through create_app(), so this must be called explicitly
logger = get_logger(__name__)


def main():
    model = build_architecture(pretrained_backbone=False)
    model.load_state_dict(torch.load(settings.MODEL_PATH, map_location="cpu"))
    model.eval()

    h, w = settings.IMG_SIZE
    example_input = torch.randn(1, 3, h, w)

    logger.info(f"Tracing {settings.ARCHITECTURE} with input shape {tuple(example_input.shape)}")
    traced = torch.jit.trace(model, example_input)
    traced.save(settings.TORCHSCRIPT_MODEL_PATH)
    logger.info(f"Saved TorchScript model to {settings.TORCHSCRIPT_MODEL_PATH}")


if __name__ == "__main__":
    main()
