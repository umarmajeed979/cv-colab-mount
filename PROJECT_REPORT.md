# Project Report — Semantic Segmentation for Autonomous Driving

A study reference for future-you. Written so that if you come back to
this months later having forgotten everything, you can read this top to
bottom and understand what was built, why every decision was made, what
broke, and what to do differently next time.

---

## 1. What this project is

A road-scene semantic segmentation system — given a photo, it colors in
every pixel by class (road, car, pedestrian, sky, etc.). Part of your
CV/ML portfolio, built on your shared `ml-project-template` boilerplate
(the "always FastAPI backend + Streamlit frontend + same folder layout"
standard you use across every portfolio project).

**Final stack:** PyTorch (DeepLabV3-ResNet50, with a from-scratch U-Net
as a swappable alternative) · CamVid dataset (11 classes) · TorchScript
export · FastAPI · Streamlit.

**Where it stands:** Full pipeline built, tested, and working end to end
— data download → preprocessing → training (Colab) → evaluation →
export → API → frontend. You trained a real model on Colab and ran it
through the frontend successfully.

---

## 2. The journey, in order

This section is a narrative — read it once to reconstruct the story,
then use Section 3 (Architecture) and Section 4 (Bugs) as reference.

1. **Scaffolded from your universal template.** Started from
   `ml-project-template`'s boilerplate zip (the one with `[never touch]`
   / `[always edit]` / `[sometimes edit]` tags per file — see your
   `CODING_STANDARDS.md` and layout doc for that contract). Deleted the
   example-only `disease_info.py`.

2. **Chose the dataset — twice.** First choice was BDD100K (10K-image
   subset, Cityscapes-style 19 classes). This turned out to be a dead
   end: the official download portal had a broken SSL cert, the
   "current" domain didn't work, and a suggested ETH Zurich mirror
   didn't even resolve on your network (`DNS_PROBE_FINISHED_NXDOMAIN`).
   After three failed attempts, switched to **CamVid** instead — hosted
   as plain files in a public GitHub repo
   (`alexgkendall/SegNet-Tutorial`), no login, no cert issues, just
   `git clone`. Smaller (701 images total vs BDD100K's 10K) but has a
   real labeled **test** split, which BDD100K doesn't. This is the
   dataset the project actually uses.

3. **Built the pipeline around CamVid's 11-class taxonomy**
   (sky/building/pole/road/pavement/tree/sign_symbol/fence/car/
   pedestrian/bicyclist + class 11 = "void/ignore"). All the
   dataset-specific code (`config.py`, `data/loader.py`,
   `data/preprocessor.py`, class list + color palette) was written
   around this.

4. **Audited `requirements.txt` for the real training environment**
   (Colab, Python 3.12, T4 GPU, PyTorch 2.11.0+cu128). Exact-pinned
   `torch`/`torchvision` as a matched pair (the one place exact pins
   matter — mismatches cause silent CUDA crashes); loosened everything
   else to floor-pinned ranges so the resolver isn't fighting Colab's
   huge pre-installed package set. `numpy` ended up **fully
   unconstrained** (no version at all) after a pinned range caused slow
   resolver backtracking on Colab — bare `numpy` just accepts whatever
   Colab already ships, which already satisfies everything else.

5. **Trained on Colab.** Hit a real hang (see Section 4, Bug #4) caused
   by `DataLoader` multiprocessing workers deadlocking — fixed by
   defaulting `--num-workers` to `0` and adding live tqdm progress bars
   so a genuine hang is now visually obvious within seconds instead of
   silent for 8+ minutes.

6. **Downloaded the trained `.pt` file and moved to local
   evaluation/deployment** (VS Code, Windows). Hit two more real bugs
   here (Section 4, Bugs #5 and #6) — both packaging/environment issues,
   not modeling issues.

7. **Ran the full local pipeline successfully**: backend (`run.py`) +
   frontend (`streamlit run frontend/app.py`), uploaded a photo, got a
   segmentation overlay back. Fixed one last bug — the frontend calling
   the wrong URL path (Section 4, Bug #7).

---

## 3. Architecture — what each file does and why

This mirrors the template's tagging convention
(`[never touch]` / `[always edit]` / `[sometimes edit]`).

### Config — the single source of truth
`app/config.py` holds everything project-specific: `CAMVID_CLASSES` (the
11 names), `CAMVID_PALETTE` (RGB color per class for visualization),
`IGNORE_INDEX=11`, `NUM_CLASSES=11`, `IMG_SIZE=(360,480)` (CamVid's
native resolution — no downsizing needed), `ARCHITECTURE` (toggle
between `"deeplabv3_resnet50"` and `"unet"`). Nothing elsewhere hardcodes
a path, class count, or threshold — that's the rule from
`CODING_STANDARDS.md` and it held up well in practice.

### Data pipeline
- `app/data/loader.py` — `CamVidSegmentationDataset` reads straight from
  the raw download (`data/raw/CamVid/<split>/`,
  `data/raw/CamVid/<split>annot/`). `ProcessedSegmentationDataset` reads
  the *resized* pairs that `preprocessor.py` writes to
  `data/processed/<split>/{images,masks}/`. Training/eval actually use
  the **processed** one. Both classes live in `app/` (not `scripts/`)
  deliberately — see Bug #5 for why that matters.
- `app/data/preprocessor.py` — pairs images with masks, resizes both
  (image bilinear, mask **nearest-neighbor** — critical, since a mask
  pixel is a class ID, not a color; blurring it with bilinear
  interpolation would invent nonsense in-between class values).

### Model
`app/core/model.py` — two architectures behind one `build_architecture()`
call:
- **DeepLabV3-ResNet50** (torchvision, ImageNet-pretrained backbone,
  swapped classifier head for 11 classes). The default — atrous
  convolutions give it a large receptive field without losing spatial
  resolution, good for road scenes where you need both large regions
  (road, sky) and thin ones (poles, signs) segmented well.
- **U-Net** (from scratch, 4-level encoder/decoder). Lighter, faster to
  iterate on. Has a size-robustness fix baked in — see Bug #1.

### Inference
`app/core/predictor.py` — the *only* place that turns raw image bytes
into a prediction. Loads the model once, runs argmax over per-pixel
logits, builds a colorized mask + an overlay blended onto the original
image, and a per-class pixel-coverage percentage list. Returns a
`PredictResponse` (defined in `app/api/schemas.py`) — this is the
"sometimes edit" file that changed from the template default, since a
segmentation response (masks + coverage) is a different shape than a
typical classification response (label + confidence).

### Evaluation
`app/core/evaluator.py` — builds a confusion matrix across the whole
val/test loader, computes per-class IoU and Dice + means. `ignore_index`
(class 11, "void") is excluded from the matrix so unlabeled pixels don't
skew the score.

### Scripts (the four pipeline stages)
- `scripts/prepare_data.py` — verify raw download exists, call
  `preprocessor.build_splits()`.
- `scripts/train_model.py` — the training loop: AdamW + cosine LR decay,
  `CrossEntropyLoss(ignore_index=11)`, saves a checkpoint only when val
  mIoU improves. Meant for Colab but runs anywhere.
- `scripts/evaluate_model.py` — loads the saved checkpoint, runs
  `evaluator.evaluate()` against the val split, prints JSON metrics.
- `scripts/export_model.py` — `torch.jit.trace` to TorchScript, for
  lighter-weight deployment.

### Frontend
`frontend/app.py` — Streamlit: upload → POST to the backend's
`/api/v1/predict` → show overlay + mask side by side + per-class
coverage bars.

### Untouched (template "never touch" files)
`app/__init__.py`, `app/api/routes.py`, `app/api/middleware.py`,
`app/utils/logger.py`, `app/services/storage.py`, `run.py`, `wsgi.py`,
`Dockerfile`, `pyproject.toml`, `tests/conftest.py` — identical to the
template, exactly as the boilerplate contract intends.

---

## 4. Bugs hit, root causes, and fixes

This is the highest-value section for next time — these are mistakes
worth recognizing faster if they show up again in a different project
built on the same template.

**Bug #1 — U-Net crashed on CamVid's exact input size.**
CamVid images are 480×360. A 4-level U-Net downsamples by 2 four times
(÷16). 480÷16=30 (fine), but 360÷16=22.5 (not an integer) — the decoder's
upsampled tensor came back a different size than the matching encoder
skip-connection, and concatenation threw a shape-mismatch error.
*Fix:* rather than constraining input size to be divisible by 16 forever,
made the U-Net robust to *any* size — pad the upsampled tensor to match
the skip connection at each level, and a final `interpolate` guard in
case the output is still a pixel off. General lesson: **don't force
"nice" input dimensions on a model — make the model handle arbitrary
dimensions**, it's more robust for whatever the next dataset's native
resolution happens to be.

**Bug #2 — `pretrained_backbone=False` downloaded weights anyway.**
`deeplabv3_resnet50(weights=weights)` — the `weights` argument controls
the *segmentation head* (COCO-pretrained, useless for a custom class
set), not the ResNet50 *backbone*. The backbone downloads via a
*separate* `weights_backbone` argument that defaults to
ImageNet-pretrained regardless of what `weights` is set to. This meant
every "load the model to run inference" call was silently trying to
download a ~100MB file it didn't need (harmless when it works, but wastes
time and fails hard with no internet). *Fix:* explicitly pass
`weights_backbone=None` when `pretrained_backbone=False`. General lesson:
**torchvision model constructors often have more than one weights
parameter — check the actual signature, don't assume one flag controls
everything.**

**Bug #3 — Broken dataset portals (BDD100K).**
Not a code bug, but cost real time: tried three separate official/mirror
URLs for BDD100K before concluding the dataset source itself was the
problem, not your network. *Lesson for next time:* when a research
dataset's official portal is flaky, check whether it's mirrored as plain
files in a GitHub repo first — those are far more reliable than
university-hosted download portals (no login walls, no custom certs, no
DNS surprises).

**Bug #4 — Colab training hung indefinitely with no output.**
`DataLoader(..., num_workers=2)` on Colab deadlocks — a well-documented
issue caused by Colab's limited `/dev/shm` (shared memory) combined with
fork-based multiprocessing after CUDA is already initialized in the main
process. It doesn't error, it just hangs forever. *Fix:* default
`--num-workers` to `0` (single-process loading — this dataset is small
enough that it costs almost nothing). Also added a `tqdm` progress bar
inside both the training and evaluation loops — previously the code only
logged once per *entire* epoch, so a genuine hang and "just running
slowly" looked identical for 8+ minutes. *Lesson:* **always have
per-batch visibility in a training loop, not just per-epoch** — silent
loops make debugging hangs much harder than it needs to be.

**Bug #5 — `ModuleNotFoundError: No module named 'scripts'`.**
`evaluate_model.py` imported a class from `scripts/train_model.py`
(`from scripts.train_model import ProcessedSegmentationDataset`). This
only works by accident, depending on exactly how/where the script is
invoked, because `scripts/` was never set up as an installed package —
only `app/` is (via `pip install -e .`). *Fix:* moved the shared
`ProcessedSegmentationDataset` class into `app/data/loader.py`, which is
*always* importable since `app/` is the one properly installed package.
*Lesson:* **never import across sibling top-level scripts — shared code
belongs inside the installed package (`app/`), not in `scripts/`.**

**Bug #6 — `ImportError` pointing at the wrong folder entirely.**
After the Bug #5 fix, still got an import error — but the traceback
showed Python importing `loader.py` from an *old, differently-named*
copy of the project folder, not the one the command was run from.
Root cause: `pip install -e .` records an *absolute path* to wherever it
was run. Copying/renaming the project folder (`_new`, `updated version`,
etc.) without redoing the editable install leaves pip still pointing at
the stale location. *Fix:*
`pip install -e . --force-reinstall --no-deps` inside the current
folder. *Lesson:* **an editable install is tied to a specific folder
path — treat renaming/copying a project folder as requiring a
reinstall, every time.**

**Bug #7 — Frontend 404 on `/predict`.**
`frontend/app.py` posted to `f"{API_URL}/predict"`, but
`app/__init__.py` registers all routes under `prefix="/api/v1"`, so the
real endpoint is `/api/v1/predict`. *Fix:* corrected the frontend's URL.
*Lesson:* **whenever the backend's route prefix changes (or is copied
from a template where you're not 100% sure what it is), grep for every
place a URL is hardcoded on the frontend side and check it matches** —
this is an easy one-line mismatch to miss because both sides "look"
correct in isolation.

**Bug #8 — `setup_logging()` crashed on a fresh Colab clone.**
`logging.FileHandler("logs/error.log")` doesn't create missing parent
directories. Worked fine locally (the `logs/` folder already existed on
disk from earlier runs) but crashed immediately on a clean `git clone`
on Colab — because git never tracked the empty `logs/` folder in the
first place (`.gitignore` excludes the `*.log` files inside it, and an
otherwise-empty directory isn't tracked by git at all). *Fix:*
`Path("logs").mkdir(parents=True, exist_ok=True)` at the top of
`setup_logging()`, plus a committed `logs/.gitkeep` so the folder is
visibly present right after cloning. *Lesson:* **any code that writes to
a folder should create that folder itself — never assume a folder
exists just because it does on your own machine.** This is also a
reminder that "works locally" and "works on a truly fresh clone" are
different tests; the second one is what actually matters for a repo
someone else (or future-you, or Colab) will clone from scratch.

---

## 5. Reusable checklist for the *next* portfolio project

Since this is built from the same shared template, these apply directly
to whatever you build next:

- [ ] After cloning a fresh copy of any project (or renaming an existing
      folder), run `pip install -e . --force-reinstall --no-deps` before
      anything else. Don't assume an old editable install still applies.
- [ ] Always `python -m pip install ...`, never bare `pip install ...`,
      especially on Windows with multiple Python installs.
- [ ] Default any Colab `DataLoader` to `num_workers=0` unless you've
      specifically confirmed higher values don't hang in that runtime.
- [ ] Add per-batch progress output (tqdm) to any training/eval loop from
      the start — don't wait until a silent hang forces you to add it.
- [ ] Before committing to a dataset with a login-gated portal, check if
      it's mirrored as plain files on GitHub first.
- [ ] When using a torchvision pretrained model constructor, check for
      *both* a `weights` and a `weights_backbone` (or similar) parameter
      — don't assume one flag disables all downloading.
- [ ] Keep `torch`/`torchvision` as an exact-pinned matched pair in
      `requirements.txt`; loosen everything else to floor-pinned ranges
      (`>=X,<Y`) so the resolver isn't fighting a big pre-installed stack
      like Colab's. Leave `numpy` fully unpinned unless you have a
      specific reason not to.
- [ ] Any class shared between two `scripts/*.py` files belongs in `app/`
      instead — `scripts/` is never an installed package.
- [ ] After changing a backend route prefix, grep the frontend for
      hardcoded URLs and confirm they still match.
- [ ] Any code that creates a file inside a folder should create that
      folder itself first (`mkdir(parents=True, exist_ok=True)`) — don't
      assume it exists just because it does on your machine. Test on a
      genuinely fresh clone before trusting it works elsewhere (e.g.
      Colab), not just locally where leftover folders from earlier runs
      can hide the bug.

---

## 6. Quick command reference

See `RUNBOOK.md` for the full step-by-step with verification checks.
Condensed version:

```powershell
# setup
python -m pip install -r requirements.txt
python -m pip install -e .
copy .env.example .env

# dataset
git clone --depth 1 https://github.com/alexgkendall/SegNet-Tutorial data/raw/_segnet_tmp
mv data/raw/_segnet_tmp/CamVid data/raw/CamVid
rm -r -force data/raw/_segnet_tmp
python scripts/prepare_data.py

# train (Colab, GPU) — num_workers 0 is now the default
python scripts/train_model.py --epochs 30 --batch-size 8 --lr 1e-4

# evaluate / export / run (local, after downloading the trained .pt)
python scripts/evaluate_model.py
python scripts/export_model.py
python run.py                     # backend, :8000
streamlit run frontend/app.py     # frontend, :8501, separate window
```
