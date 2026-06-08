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
- Weights live in `weights/`, are **not** committed (Ultralytics AGPL licensing caution),
  and are produced by the training scripts — they are our "current best models".
  Present: `best_yolo_classifier.pt`, `best_yolo_extractor.pt`.
- Don't add weight-sync/fetch scripts; regenerate via `scripts/bin/train_*.sh`.

## Environment gotchas (macOS, verified 2026-06-08)
- **cairo**: installed via brew (`cairo` 1.18.4) but Python's `find_library` doesn't search
  `/opt/homebrew/lib`. Export `DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib` before any
  command that imports `cairosvg` (eval / process-new-raw / render paths).
- **3LC from source = no real key needed.** The frozen wheel hard-validates the key at import
  (ignores `TLC_DISABLE_ACCOUNT_SERVICE`). The **source** install honors it, so the dev setup is:
  ```bash
  uv sync                                              # public releases (keep pyproject as-is)
  uv pip install -e ../tlc-monorepo -e ../3lc-ultralytics   # editable source overlay
  ```
  Then with `.env` holding `TLC_DISABLE_ACCOUNT_SERVICE=1` and a throwaway `TLC_API_KEY`, the
  **full** suite + eval + 3LC paths run locally with no real key. We use `3lc-ultralytics`
  (`tlc_ultralytics`), **not** bare ultralytics — bare loses the 3LC integration.
- `pyproject.toml` stays pointed at the public PyPI/cloudrepo releases so external users can
  `uv sync` and go. The editable installs are a local overlay only — don't commit them.
- **Bare-ultralytics is opt-in only**, for the no-source/no-key case (e.g. external CI):
  `CHESSVISION_ALLOW_BARE_ULTRALYTICS=1`. The YOLO loaders (`_import_yolo` in `utils.py`)
  otherwise fail loud if `tlc_ultralytics` can't import.
- `.env` is **not** auto-loaded by pytest/CLI — export it first: `set -a; source .env; set +a`.

## Commands (use the venv)
```bash
source .venv/bin/activate
set -a; source .env; set +a                          # TLC_DISABLE_ACCOUNT_SERVICE=1 + throwaway key
export DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib   # for cairo-using paths

# Full suite — works locally with source installs, no real key
python -m pytest tests/ -q
./scripts/bin/evaluate.sh
./scripts/bin/train_yolo_classifier.sh
./scripts/bin/train_yolo_board_extractor.sh

# No source/key at all (external CI smoke test of the CV path only):
CHESSVISION_ALLOW_BARE_ULTRALYTICS=1 python -m pytest tests/test_chessvision.py --no-cov -q

# Lint
ruff check chessvision/
```

## Conventions
- Python ≥3.10, ruff (line length 120) + mypy configured in `pyproject.toml`.
- `chessvision/pytorch_unet/` is a submodule (third-party) — excluded from lint/type/cov.
- Keep `tlc` usage out of the core `chessvision/` package; it belongs in `scripts/`.

## Running on the godfire GPU node (Linux, RTX 4080)

This node piggybacks on the parent uv workspace at `/home/gubbis/projects/`, which holds the
`3lc-ultralytics` and `tlc-monorepo` editable sources.

- **Do NOT `uv sync` inside this dir** — `pyproject.toml` path-deps to `../3lc-ultralytics` /
  `../tlc-monorepo` collide with the parent workspace ("Nested workspaces are not supported").
- Use the parent venv python directly: `/home/gubbis/projects/.venv/bin/python`. `chessvision`
  is installed editable there via `VIRTUAL_ENV=/home/gubbis/projects/.venv uv pip install -e . --no-deps`.
- TLC env (`TLC_API_KEY=1`, `TLC_DISABLE_ACCOUNT_SERVICE=1`) comes from the parent `.envrc`
  (direnv `source_up`); no `3lc login` needed. tlc runs from `../tlc-monorepo` source.
- 3LC writes runs/tables under `/home/gubbis/.local/share/3LC/projects/`.
- The `scripts/bin/*.sh` wrappers use `uv run`, which fails here — call the venv python directly.
- Init the UNet submodule once: `git submodule update --init`.
