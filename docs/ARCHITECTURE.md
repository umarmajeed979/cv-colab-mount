# Architecture

## Pipeline
1. `app/data/loader.py` / `preprocessor.py` — pair CamVid images with
   their `*annot` masks, resize to `IMG_SIZE` (mask via nearest-neighbor
   so label ids are never blurred), write to `data/processed/`.
2. `app/core/model.py` — builds either `deeplabv3_resnet50` (torchvision,
   ImageNet-pretrained backbone, replaced classifier head for 11 classes)
   or a from-scratch 4-level `UNet`, selected via `settings.ARCHITECTURE`.
3. `scripts/train_model.py` — trains on Colab GPU (or locally — CamVid is
   small enough), CrossEntropyLoss with `ignore_index=11`, AdamW + cosine
   LR schedule, checkpoints on best val mIoU.
4. `app/core/predictor.py` — loads the trained weights once at startup,
   runs argmax over per-pixel logits, builds a colorized mask + overlay,
   and a per-class pixel-coverage summary.
5. `app/api/routes.py` (untouched) exposes `/health` and `/predict`;
   `frontend/app.py` (Streamlit) is a thin client over that API.

## Class taxonomy
11 classes, SegNet/CamVid ids (0-10), `11` = ignore/unlabelled (void).
Full list + RGB palette live in `app/config.py` (`CAMVID_CLASSES`,
`CAMVID_PALETTE`).

## Why DeepLabV3 as the default
Atrous/dilated convolutions give DeepLabV3 a large receptive field without
losing spatial resolution — a good fit for road scenes where both large
regions (road, sky) and small/thin ones (poles, traffic signs) need to be
segmented accurately. U-Net is kept as a lighter, faster-to-train
alternative (`ARCHITECTURE=unet` in `.env`) for quicker iteration or
lower-resource fine-tuning.
