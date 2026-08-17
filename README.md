# Semantic Segmentation for Autonomous Driving

Road scene semantic segmentation — identifies road, vehicles, pedestrians,
sign symbols, and 6 other classes per pixel, using the 11-class SegNet
taxonomy on the CamVid dataset. Built for the self-driving-perception
slice of the portfolio.

## Stack
PyTorch (DeepLabV3-ResNet50 / U-Net) · CamVid · TorchScript · FastAPI · Streamlit
Skills demonstrated: segmentation, autonomous systems, performance optimization

## Setup
```bash
pip install -r requirements.txt
pip install -e .
cp .env.example .env
python run.py            # backend on :8000
streamlit run frontend/app.py   # frontend on :8501, separate terminal
```

## Dataset — CamVid (free, no registration, plain GitHub files)
No login, no cert issues, no request forms — the dataset is just PNG files
in a public repo (`alexgkendall/SegNet-Tutorial`), the standard SegNet
11-class CamVid split, images already at 480x360 with pixel-value class-id
masks (no color decoding needed) and — unlike most driving segmentation
sets — a real labeled **test** split too, not just train/val.

```bash
git clone --depth 1 https://github.com/alexgkendall/SegNet-Tutorial data/raw/_segnet_tmp
mv data/raw/_segnet_tmp/CamVid data/raw/CamVid
rm -rf data/raw/_segnet_tmp
```
This gives you:
```
data/raw/CamVid/{train,val,test}/*.png        # 480x360 RGB images
data/raw/CamVid/{train,val,test}annot/*.png   # matching single-channel masks, values 0-10 = class id, 11 = void
```
Then build the processed split:
```bash
python scripts/prepare_data.py
```
367 train / 101 val / 233 test images — small enough to do a full first
training pass locally before ever touching Colab, if you want a fast
sanity check.

## Training (Google Colab, GPU)
```python
!git clone <your-repo-url>
%cd <repo>
!pip install -r requirements.txt -q
!pip install -e . -q
# upload data/raw/CamVid/ via Drive mount or direct upload, then:
!python scripts/prepare_data.py
!python scripts/train_model.py --epochs 30 --batch-size 8 --lr 1e-4
```
Checkpoints save to `data/models/model_final.pt` whenever val mIoU improves.
Download that file back into your local `data/models/` before running the API.

## Evaluation
```bash
python scripts/evaluate_model.py
```
Prints mean IoU / mean Dice and a per-class breakdown against the val split.

## Export (TorchScript, for edge/deployment)
```bash
python scripts/export_model.py
```
Writes `data/models/model_final.torchscript.pt`.

## API
`POST /api/v1/predict` — multipart image upload → JSON with:
- `overlay_image_base64` — original image with the colorized mask blended in
- `mask_image_base64` — colorized mask only
- `class_coverage` — % of image pixels per detected class, sorted desc
- `inference_time_ms`, `model_version`

`GET /api/v1/health` — `{status, model_loaded}`

## Frontend
`streamlit run frontend/app.py` — upload an image, see the overlay + mask
side by side and the per-class coverage breakdown.

## What changes per project
Edit these — this is where a project's actual identity lives:
- `app/config.py` — PROJECT_NAME, DESCRIPTION, MODEL_PATH, class list, IMG_SIZE
- `app/core/model.py` — architecture definitions (DeepLabV3 / U-Net)
- `app/core/predictor.py` — segmentation inference + overlay/coverage building
- `app/data/preprocessor.py` and `app/data/loader.py` — CamVid-specific loading
- `app/utils/image_utils.py` — resize/normalize + mask↔color helpers
- `frontend/app.py` — upload/results UI
- `requirements.txt` — torch/torchvision/streamlit added for this project
- `.env`, `README.md` — project-specific values and docs

Sometimes edit:
- `app/api/schemas.py` — changed here: segmentation response (masks +
  class coverage) instead of a flat label/confidence list
- `app/utils/validators.py` — left at template default (image type check only)

Leave alone — identical across every project, don't touch:
- `app/__init__.py`, `app/utils/logger.py`, `app/api/routes.py`,
  `app/api/middleware.py`, `app/services/storage.py`, `run.py`,
  `wsgi.py`, `Dockerfile`, `pyproject.toml`, `tests/conftest.py`
