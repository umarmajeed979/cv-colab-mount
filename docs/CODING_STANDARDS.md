# Coding standards

Applies to every project in the portfolio. Distilled from Zero Point —
this file, not the manual, is what gets attached/copied per project.

## 1. Install the project as a package, don't fight the path
Every project has a `pyproject.toml`. After cloning:
    pip install -e .
This installs the repo in "editable" mode — `app/` becomes a real
importable package (`from app.core.model import load_model` works
from anywhere: scripts, tests, a notebook) without sys.path hacks or
relative-import errors. Do this once per project, right after
`pip install -r requirements.txt`, before running anything else.

## 2. One place for every kind of thing
- Config values (paths, thresholds, model version) → `config.py` only.
  Never hardcode a path or magic number inside a route or a script.
- Logging → `get_logger(__name__)` from `utils/logger.py`. Never `print()`.
- Request/response shape → `api/schemas.py` only.
- Inference logic → `core/predictor.py` only. Routes call it, never
  reimplement it.

## 3. Docstrings
Every module gets a one-paragraph docstring at the top explaining
what it does and, if relevant, what in it changes per-project vs.
what stays fixed (see the template files for the pattern).
Every public function gets a one-line docstring — what it does, not
how (the code shows how).

## 4. Errors
Catch specific exceptions, not bare `except:`. Log with
`logger.exception(...)` before re-raising or returning an HTTP error
so the traceback isn't lost. Never fail silently.

## 5. Naming
snake_case for functions/variables, PascalCase for classes,
UPPER_CASE for constants. File names match their primary class/
function where there's one (`predictor.py` → `Predictor`).

## 6. Commits
Small, one logical change each. Message format:
    <area>: <what changed>
e.g. `predictor: add confidence thresholding`, `frontend: wire up
file upload to /predict`.

## 7. What's allowed to differ per project
See README.md → "What changes per project" for the exact file list.
Anything not on that list should be identical across every repo in
the portfolio — if you find yourself editing it, that's a signal to
fix the template instead of patching one project.
