"""
Model loading and lifecycle. Two architectures are supported behind
settings.ARCHITECTURE ("deeplabv3_resnet50" or "unet") -- predictor.py
never needs to know which one is active, it just calls get_model().
"""
import torch
import torch.nn as nn

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

_model = None


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


def _pad_to_match(upsampled: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
    """
    Zero-pad `upsampled` so its H/W match `skip`'s. Needed because
    transpose-conv upsampling doesn't exactly invert max-pool when an
    input dimension isn't divisible by 2**num_pool_layers (e.g. CamVid's
    360px height isn't divisible by 16) -- without this, concatenation
    in forward() would throw a shape mismatch.
    """
    diff_h = skip.size(2) - upsampled.size(2)
    diff_w = skip.size(3) - upsampled.size(3)
    if diff_h == 0 and diff_w == 0:
        return upsampled
    return nn.functional.pad(
        upsampled,
        [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2],
    )


class UNet(nn.Module):
    """Standard 4-level U-Net, used as the lighter-weight alternative to DeepLabV3."""

    def __init__(self, num_classes: int, base_ch: int = 64):
        super().__init__()
        chs = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8]

        self.down1 = DoubleConv(3, chs[0])
        self.down2 = DoubleConv(chs[0], chs[1])
        self.down3 = DoubleConv(chs[1], chs[2])
        self.down4 = DoubleConv(chs[2], chs[3])
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(chs[3], chs[3] * 2)

        self.up4 = nn.ConvTranspose2d(chs[3] * 2, chs[3], 2, stride=2)
        self.upconv4 = DoubleConv(chs[3] * 2, chs[3])
        self.up3 = nn.ConvTranspose2d(chs[3], chs[2], 2, stride=2)
        self.upconv3 = DoubleConv(chs[2] * 2, chs[2])
        self.up2 = nn.ConvTranspose2d(chs[2], chs[1], 2, stride=2)
        self.upconv2 = DoubleConv(chs[1] * 2, chs[1])
        self.up1 = nn.ConvTranspose2d(chs[1], chs[0], 2, stride=2)
        self.upconv1 = DoubleConv(chs[0] * 2, chs[0])

        self.out_conv = nn.Conv2d(chs[0], num_classes, 1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool(d1))
        d3 = self.down3(self.pool(d2))
        d4 = self.down4(self.pool(d3))
        b = self.bottleneck(self.pool(d4))

        u4 = self.upconv4(torch.cat([_pad_to_match(self.up4(b), d4), d4], dim=1))
        u3 = self.upconv3(torch.cat([_pad_to_match(self.up3(u4), d3), d3], dim=1))
        u2 = self.upconv2(torch.cat([_pad_to_match(self.up2(u3), d2), d2], dim=1))
        u1 = self.upconv1(torch.cat([_pad_to_match(self.up1(u2), d1), d1], dim=1))

        out = self.out_conv(u1)
        # Guard: if input H/W weren't divisible by 16, output can end up a few
        # px off from the input size even after padding above -- interpolate
        # to force an exact match, since predictor.py/evaluator.py assume
        # output shape == input shape.
        if out.shape[-2:] != x.shape[-2:]:
            out = nn.functional.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)

        return {"out": out}  # dict output to match torchvision's DeepLabV3 contract


def _build_deeplabv3(num_classes: int, pretrained_backbone: bool = True) -> nn.Module:
    from torchvision.models.segmentation import deeplabv3_resnet50
    from torchvision.models import ResNet50_Weights

    # weights=None always (segmentation head is COCO/21-class, useless for us);
    # weights_backbone controls just the ImageNet-pretrained ResNet50 trunk, and
    # must be explicitly None when loading our own trained state_dict -- otherwise
    # torchvision downloads the ImageNet backbone even though we're about to
    # overwrite every weight with load_state_dict() right after.
    weights_backbone = ResNet50_Weights.DEFAULT if pretrained_backbone else None
    model = deeplabv3_resnet50(weights=None, weights_backbone=weights_backbone)
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    if model.aux_classifier is not None:
        model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    return model


def build_architecture(pretrained_backbone: bool = True) -> nn.Module:
    """Construct a fresh (untrained-head) model for the configured architecture. Used by train_model.py."""
    if settings.ARCHITECTURE == "deeplabv3_resnet50":
        return _build_deeplabv3(settings.NUM_CLASSES, pretrained_backbone)
    elif settings.ARCHITECTURE == "unet":
        return UNet(settings.NUM_CLASSES)
    raise ValueError(f"Unknown ARCHITECTURE: {settings.ARCHITECTURE!r}")


def load_model():
    """Load trained weights once, cache in module state. Called at app startup."""
    global _model
    if _model is None:
        logger.info(f"Loading {settings.ARCHITECTURE} model from {settings.MODEL_PATH}")
        try:
            model = build_architecture(pretrained_backbone=False)
            state_dict = torch.load(settings.MODEL_PATH, map_location=settings.DEVICE)
            model.load_state_dict(state_dict)
            model.to(settings.DEVICE)
            model.eval()
            _model = model
        except FileNotFoundError:
            logger.warning(f"No weights found at {settings.MODEL_PATH} -- train first via scripts/train_model.py")
    return _model


def get_model():
    return _model if _model is not None else load_model()
