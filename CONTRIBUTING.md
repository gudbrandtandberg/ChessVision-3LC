# Contributing to ChessVision

Thanks for your interest! This guide covers the dev environment in tiers — pick the one that
matches your access. The **default path works for everyone with a free 3LC account**; the
other tiers exist for narrower needs.

## Setup tiers

| Tier | Who it's for | Install | Auth | Can run |
|---|---|---|---|---|
| **1 — Public (default)** | anyone | `uv sync` (public wheels) | free 3LC key, account service **on** | everything: inference, training, eval, 3LC Dashboard |
| **2 — Source overlay** | 3LC devs with repo access | wheels + editable `tlc` / `tlc_ultralytics` | throwaway key + account service **off** | everything, **fully offline** (no real key) |
| **3 — Bare ultralytics** | offline CI, no 3LC at all | `uv sync` | none | **inference only** (no training / 3LC) |

The only reason to use Tier 2 is running with **account service disabled** (`TLC_DISABLE_ACCOUNT_SERVICE=1`):
the frozen public `3lc` wheel hard-validates the key at import and ignores that flag, while the
**source** install honors it. Everyone else — including external contributors who want to train —
just uses Tier 1 with a free key.

> Maintainer note: Tier 2 (source) is the day-to-day internal path, but **Tier 1 (wheels + real
> key) is what every external user actually hits** and is easy to let rot. Periodically run the
> suite against a clean `uv sync` (no overlay) + a real key to keep the default experience honest.

---

## Tier 1 — Public (default)

See the [README installation section](README.md#installation). In short:

```bash
git clone https://github.com/gudbrandtandberg/ChessVision-3LC.git
cd ChessVision-3LC
git submodule update --init        # UNet submodule (only needed for the UNet fallback path)
uv sync --all-extras               # public deps, incl. 3lc-ultralytics + 3lc from the public index
3lc login <your-free-key>          # from account.3lc.ai
```

The trained YOLO weights are committed in `weights/`, so inference and the test suite work
immediately:

```bash
pytest tests/
```

## Tier 2 — Source overlay (fully offline, 3LC devs)

For contributors with access to the `tlc` monorepo and `3lc-ultralytics` sources. Layer editable
installs over the public sync so account service can be disabled (no real key needed):

```bash
uv sync                                                   # public releases (keep pyproject as-is)
uv pip install -e ../tlc-monorepo -e ../3lc-ultralytics   # editable source overlay
```

The source overlay is **local only — don't commit it.** `pyproject.toml` stays pointed at the
public 3LC index so external `uv sync` keeps working. The overlay tracks the `tlc` dev line, which
is API-compatible with the public 3.0 wheel.

Then put a throwaway key + the disable flag in `.env` (not auto-loaded — export it yourself):

```bash
# .env
TLC_DISABLE_ACCOUNT_SERVICE=1
TLC_API_KEY=throwaway

set -a; source .env; set +a
pytest tests/        # full suite + eval + 3LC paths, no real key
```

We use `3lc-ultralytics` (`tlc_ultralytics`), **not** bare ultralytics — bare loses the 3LC
integration.

## Tier 3 — Bare ultralytics (inference only)

For environments with no 3LC source and no key (e.g. external CI smoke tests). Opt in explicitly;
the YOLO loaders otherwise fail loud if `tlc_ultralytics` can't import:

```bash
CHESSVISION_ALLOW_BARE_ULTRALYTICS=1 python -m pytest tests/test_chessvision.py --no-cov -q
```

Training stays coupled to `tlc_ultralytics` — this tier is inference only.

---

## Platform notes

### macOS (Apple Silicon, MPS)

- **cairo**: installed via brew (`cairo` 1.18.4) but Python's `find_library` doesn't search
  `/opt/homebrew/lib`. Export `DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib` before any command
  that imports `cairosvg` (eval / process-new-raw / render paths).
- Training/inference run on MPS; `channels_last` memory format is skipped on MPS (CPU/CUDA only).

### Linux + CUDA (e.g. an RTX 4080 node)

- Use the project's **own standalone `.venv`** (`uv venv` in the project), not a shared parent
  workspace venv.
- 3LC writes runs/tables under `~/.local/share/3LC/projects/`.
- Long training: run `nohup`'d and poll the log. Do **not** write a waiter that greps its own
  target string — `pgrep -f train_yolo_classifier` matches the waiter itself and loops forever.

---

## Conventions

- Python ≥3.10; `ruff` (line length 120) + `mypy` configured in `pyproject.toml`.
- `chessvision/pytorch_unet/` is a third-party submodule — excluded from lint/type/cov.
- Keep `tlc` usage **out of** the core `chessvision/` package; it belongs in `scripts/`.
- Lint before pushing: `ruff check chessvision/`.

Direction and priorities live in [`ROADMAP.md`](ROADMAP.md).
