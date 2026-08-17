# Guide — Uploading This Project to GitHub

Commands are PowerShell, matching your setup. Run these from inside the
project root (`...\semantic-segmentation-driving>`).

---

## 1. Create the repository on GitHub

1. Go to https://github.com/new
2. Repo name: `semantic-segmentation-driving` (or whatever you prefer)
3. Leave it **empty** — do NOT check "Add a README", "Add .gitignore", or
   "Add a license". You already have all of these locally; checking
   those boxes creates files on GitHub that conflict with your local
   ones and forces an extra merge step for no reason.
4. Click **Create repository**. You'll land on a page with a URL like
   `https://github.com/<your-username>/semantic-segmentation-driving.git`
   — copy that, you need it in step 3.

---

## 2. Check what's about to be uploaded

Your project already has a `.gitignore` that excludes the heavy/private
stuff (`data/raw/`, `data/processed/`, `data/models/*.pt`, `.env`,
`venv/`, `__pycache__/`, logs). Verify it's doing its job before you
commit anything:

```powershell
git init
git add .
git status
```

Read the list `git status` prints. You should **not** see anything under
`data/raw/CamVid/`, `data/models/model_final.pt`, or your `.env` file —
if any of those show up, stop and check your `.gitignore` before
continuing (paste me the `git status` output if something looks wrong).

This matters for two reasons: your trained weights file and the CamVid
dataset are hundreds of MB — way too large for a normal GitHub repo —
and `.env` could contain machine-specific settings you don't want public.

---

## 3. Connect to GitHub and push

```powershell
git branch -M main
git remote add origin https://github.com/<your-username>/semantic-segmentation-driving.git
git commit -m "Initial commit: semantic segmentation pipeline"
git push -u origin main
```

Replace `<your-username>` with your actual GitHub username (from the URL
you copied in step 1).

**If `git commit` says "Please tell me who you are"** — run these once
(one-time machine setup, not per-project):
```powershell
git config --global user.name "Your Name"
git config --global user.email "your-github-email@example.com"
```
then re-run the `git commit` line above.

---

## 4. Authentication (the part that usually trips people up)

GitHub no longer accepts your account password for `git push` over
HTTPS. When the push prompts for credentials, you need a **Personal
Access Token** instead:

1. Go to https://github.com/settings/tokens → **Generate new token**
   (classic is simplest) → check the `repo` scope → generate.
2. Copy the token (you only see it once — save it somewhere safe).
3. When `git push` prompts for a username/password, enter your GitHub
   username, and **paste the token as the password**.

Windows will usually offer to remember this via Credential Manager after
the first successful push, so you shouldn't have to repeat this every
time.

**Alternative, if you'd rather skip tokens entirely:** install
[GitHub Desktop](https://desktop.github.com/) — sign in through the app,
then `git push` from PowerShell will just work using that saved login.

---

## 5. Verify it worked

Refresh the GitHub repo page in your browser. You should see your
folders (`app/`, `scripts/`, `frontend/`, etc.) and files
(`README.md`, `RUNBOOK.md`, `PROJECT_REPORT.md`, `requirements.txt`,
...) — but **not** `data/raw/`, `data/processed/`, or
`data/models/model_final.pt`.

---

## 6. Making future changes

Once it's up, the day-to-day workflow for any edit is:

```powershell
git add .
git commit -m "short description of what changed"
git push
```

---

## 7. Using this repo in Colab

Now that it's on GitHub, `RUNBOOK.md` step 6 (Colab training) works with
your real URL instead of a placeholder:

```python
!git clone https://github.com/<your-username>/semantic-segmentation-driving.git
%cd semantic-segmentation-driving
```

If the repo is **private**, `git clone` in Colab will prompt for
authentication the same way `git push` did — same Personal Access Token
works there too (paste it as the password when prompted).
