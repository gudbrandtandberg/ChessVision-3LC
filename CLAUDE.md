# CLAUDE.md — working notes for this repo

Project map and gotchas for Claude (and humans). Keep terse and current.
Direction & roadmap live in `ROADMAP.md`.

## What this is
Image → FEN chess-position pipeline, and a 3LC platform demo/playground.
Pipeline: board segmentation (YOLO-seg) → contour/perspective extraction → square
classification (YOLO-cls) → chess-rule validation. Entry point: `chessvision.ChessVision`
in `chessvision/core.py`.

## Models / weights
- **YOLO is the default and preferred path** for both board extraction and classification.
  UNet remains as a fallback only if `ultralytics` is unavailable.
- The two best models **are committed** in `weights/` (`best_yolo_classifier.pt`,
  `best_yolo_extractor.pt`) — the repo is AGPL-3.0, so the weights ship in-repo (see the
  ROADMAP decision log). They're produced by the training scripts — our "current best models".
- Don't add weight-sync/fetch scripts; regenerate via `scripts/bin/train_*.sh`.

## Setup → see CONTRIBUTING.md
Full environment setup (the three tiers + macOS / Linux-CUDA platform notes) lives in
[`CONTRIBUTING.md`](CONTRIBUTING.md). The dev default here is **Tier 2 (source overlay)**:
`uv sync` + editable `../tlc-monorepo` / `../3lc-ultralytics`, with `.env` holding
`TLC_DISABLE_ACCOUNT_SERVICE=1` + a throwaway key so the full suite runs offline (the public
wheel ignores that flag and 403s; the source honors it). Overlay is local-only — don't commit it.

## Gotchas Claude trips on
- `.env` is **not** auto-loaded by pytest/CLI — export it first: `set -a; source .env; set +a`.
- macOS cairo: `export DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib` before any `cairosvg` path
  (eval / process-new-raw / render).
- YOLO loaders (`_import_yolo` in `utils.py`) fail loud if `tlc_ultralytics` is missing; bare
  ultralytics is opt-in via `CHESSVISION_ALLOW_BARE_ULTRALYTICS=1` (inference only — training
  stays 3LC-coupled).
- godfire: prefer `.venv/bin/python` over the `scripts/bin/*.sh` wrappers (they `uv run`); 3LC
  writes under `~/.local/share/3LC/projects/`; pre-retrain weights backed up in `weights/backup/`.
- Long training: run nohup'd and poll the log; do NOT write a waiter that greps its own target
  string (`pgrep -f train_yolo_classifier` matches the waiter itself and loops forever).

## Commands (Tier 2 dev default)
```bash
source .venv/bin/activate
set -a; source .env; set +a                          # TLC_DISABLE_ACCOUNT_SERVICE=1 + throwaway key
export DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib   # macOS, for cairo-using paths

python -m pytest tests/ -q
./scripts/bin/evaluate.sh
./scripts/bin/train_yolo_classifier.sh
./scripts/bin/train_yolo_board_extractor.sh

# Offline CV-only smoke test (no source/key):
CHESSVISION_ALLOW_BARE_ULTRALYTICS=1 python -m pytest tests/test_chessvision.py --no-cov -q

ruff check chessvision/
```

## Conventions
- Python ≥3.10, ruff (line length 120) + mypy configured in `pyproject.toml`.
- `chessvision/pytorch_unet/` is a submodule (third-party) — excluded from lint/type/cov.
- Keep `tlc` usage out of the core `chessvision/` package; it belongs in `scripts/`.
