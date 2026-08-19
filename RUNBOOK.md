# Runbook — Semantic Segmentation for Autonomous Driving

Commands are for PowerShell (matches`PS E:\...>` prompt).

---

## 0. One-time check: is the right Python running?

```powershell
python --version
python -m pip --version
```

Both should print without error, and ideally the same Python folder path.
From now on, **always use `python -m pip install ...`**, never bare `pip
install ...` — this is what caused the `ModuleNotFoundError: No module
named 'torch'` error you just hit. `python -m pip` guarantees packages
install into the exact Python that will run your scripts; bare `pip` can
silently install into a *different* Python if you have more than one on
your machine (very common on Windows).

**If ever copy/rename the project folder** (e.g. `..._new`,
`...updated version`), re-run `python -m pip install -e . --force-reinstall --no-deps`
inside the *new* folder. `-e .` records an absolute path to wherever it
was run — an old install left pointing at a renamed/moved/deleted folder
causes confusing `ImportError`s that look like a code bug but aren't.

---

## 1. Install dependencies

From the project root (`E:\semantic-segmentation-driving>`):

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
```

`-e .` installs your own `app/` folder as an importable package — without
it, things like `from app.config import settings` break outside the
project root.

**Verify it worked:**
```powershell
python -c "import torch; print(torch.__version__)"
python -c "import fastapi, streamlit; print('ok')"
```
If any of these error, the install above didn't finish or didn't target
the right Python — re-run step 0, then step 1 again, and paste me the
exact error if it persists.

---

## 2. Set up `.env`

```powershell
copy .env.example .env
```
Leave it as-is for now — the defaults are already correct for this
project.

---

## 3. Download the dataset (CamVid)

```powershell
git clone --depth 1 https://github.com/alexgkendall/SegNet-Tutorial data/raw/_segnet_tmp
mv data/raw/_segnet_tmp/CamVid data/raw/CamVid
rm -r -force data/raw/_segnet_tmp
```

**Verify it worked:**
```powershell
dir data/raw/CamVid
```
You should see six folders: `train`, `trainannot`, `val`, `valannot`,
`test`, `testannot`.

If `git` isn't recognized, install Git for Windows first
(https://git-scm.com/download/win), then reopen PowerShell and retry.

---

## 4. Build the processed (resized) dataset

```powershell
python scripts/prepare_data.py
```

**Verify it worked:** it will print something like
`train -> train: wrote 367 image/mask pairs to data/processed/train`
(and the same for `val`/`validation`, `test`). Check:
```powershell
dir data/processed/train/images
```
should list ~367 `.png` files.

---

## 5. Quick local sanity-check training run (optional but recommended)

Before burning time on Colab, run a **tiny** run locally just to prove
the pipeline works end to end — 1 epoch, small batch:

```powershell
python scripts/train_model.py --epochs 1 --batch-size 2 --num-workers 0
```

This will be slow and the model will be bad after 1 epoch — that's
expected, you're only checking that it *runs without crashing* and that
loss/mIoU numbers print out. If it completes and prints a loss + mIoU
per epoch, you're good to move to Colab for the real run.

If your machine has too little RAM and it crashes/freezes, that's fine
— skip straight to Colab (step 6), which has more memory and a GPU.

---

## 6. Real training on Google Colab (GPU)

1. Pushing project to a GitHub repo (or zip it and upload to Colab
   directly / via Google Drive).
2. Open a new Colab notebook, set **Runtime → Change runtime type → GPU**.
3. Run:

```python
!git clone <your-repo-url>
%cd semantic-segmentation-driving   # or whatever your folder is named
!pip install -r requirements.txt -q
!pip install -e . -q
```

4. Get the CamVid data into Colab. Easiest: repeat the same clone command
   from step 3 above, but as a Colab cell:
```python
!git clone --depth 1 https://github.com/alexgkendall/SegNet-Tutorial data/raw/_segnet_tmp
!mv data/raw/_segnet_tmp/CamVid data/raw/CamVid
!rm -rf data/raw/_segnet_tmp
!python scripts/prepare_data.py
```

5. Train for real:
```python
!python scripts/train_model.py --epochs 30 --batch-size 8 --lr 1e-4
```
This saves the best checkpoint to `data/models/model_final.pt` as it
trains (only overwritten when validation mIoU improves). It also writes
**`logs/training_history.csv`** (one row per epoch: loss, val mIoU, val
mDice) and **`logs/training_curves.png`** (a loss/mIoU chart), both
updated after every single epoch — not just at the end.

**Important — Colab's disk disappears when the runtime disconnects.**
Everything under `logs/` and `data/models/` only exists on Colab's local
VM disk. If the runtime disconnects (timeout, crash, closing the tab),
you lose all of it unless you've downloaded it first. Two options:
- **Simplest:** download the three files you actually need right after
  training finishes (step 6 below covers the model; do the same for the
  two log files).
- **More resilient for long runs:** mount Google Drive at the start of
  the notebook and point the project there instead of `/content`, so
  files persist across disconnects automatically:
  ```python
  from google.colab import drive
  drive.mount('/content/drive')
  # then work inside /content/drive/MyDrive/semantic-segmentation-driving/
  ```

6. **Download the trained weights (and training history) back to your
   local machine:**
```python
from google.colab import files
files.download('data/models/model_final.pt')
files.download('logs/training_history.csv')
files.download('logs/training_curves.png')
```
Save these into your local project at the same relative paths
(`data/models/model_final.pt`, `logs/training_history.csv`,
`logs/training_curves.png`) so `evaluate_model.py` and your own records
match what actually happened on Colab.

---

## 7. Evaluate the trained model

Back on your local machine (or still in Colab), with the trained
`model_final.pt` in place:

```powershell
python scripts/evaluate_model.py
```

**Verify it worked:** prints a JSON block with `mean_iou`, `mean_dice`,
and a per-class breakdown for all 11 CamVid classes. It also appends
that result (with a timestamp) to `logs/eval_history.jsonl` — run this
again after retraining and you'll have every run's numbers side by side
instead of only whatever's still in your terminal scrollback.

---

## 8. (Optional) Export to TorchScript

For faster/lighter deployment inference:
```powershell
python scripts/export_model.py
```
Writes `data/models/model_final.torchscript.pt`.

---

## 9. Run the backend API

```powershell
python run.py
```
**Verify it worked:** open http://localhost:8000/api/v1/health in a
browser — you should see `{"status":"ok","model_loaded":true}`.
If `model_loaded` is `false`, double check `data/models/model_final.pt`
exists at that exact path.

---

## 10. Run the frontend (separate PowerShell window, keep step 9 running)

```powershell
streamlit run frontend/app.py
```
This opens a browser tab at http://localhost:8501. Upload a road photo,
click "Run segmentation", and you should see the colorized mask +
overlay + per-class coverage.

---

## Troubleshooting quick reference

| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'torch'` (or any package) | `pip` installed into a different Python than the one running the script | Re-run with `python -m pip install -r requirements.txt` |
| `git` not recognized | Git not installed | Install Git for Windows, reopen PowerShell |
| `data/raw/CamVid` not found when running `prepare_data.py` | Step 3 didn't complete | Re-run step 3, check `dir data/raw/CamVid` shows the 6 folders |
| Training freezes / crashes locally | Not enough RAM for a full epoch | Reduce `--batch-size` to 1 or 2, or skip to Colab (step 6) |
| Colab training cell hangs forever with no output after the pretrained-weights download line | `DataLoader` worker processes deadlocking — a well-known Colab issue when `--num-workers` > 0 (limited `/dev/shm`) | Interrupt the cell, re-run with `--num-workers 0` (this is now the script's default). If it's genuinely still running rather than stuck, each batch now prints a progress bar (tqdm) so you can tell the difference at a glance instead of waiting 8+ minutes for the next log line |
| `model_loaded: false` on `/health` | No weights at `data/models/model_final.pt` | Finish training (step 6) and place the file at that exact path |
| Streamlit page loads but predictions fail | Backend (step 9) isn't running | Make sure `python run.py` is running in another window first |
| `ImportError: cannot import name 'X' ... Did you mean 'Y'` pointing at a path in a *different* folder than the one you're running from | Stale editable install (`pip install -e .`) still pointing at an old/renamed copy of the project | Re-run `python -m pip install -e . --force-reinstall --no-deps` from inside your current folder |
| Frontend shows `Request to backend failed: 404 ... /predict` | Frontend called `/predict` but routes live under `/api/v1/predict` (see `app/__init__.py`'s router prefix) | Already fixed in `frontend/app.py` in this version — if you ever change the router prefix in `app/__init__.py`, update `frontend/app.py`'s request URL to match |
