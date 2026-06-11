"""ChessVision Test Set — a 3LC Insights (compute-service) plugin.

Wraps the "add a curated test-data batch" workflow as a hub sidebar plugin:

    sample candidates → model-assisted draft FENs → human review → commit to 3LC tables

Design: this plugin is a **thin orchestrator**. The heavy stages (sampling the LaCie upload
pool, running the YOLO pipeline, writing 3LC tables) are run by shelling into the ChessVision
repo's own ``.venv`` — so the compute-service venv stays clean and we reuse ChessVision's exact
dependencies (torch/ultralytics/tlc). The lightweight review endpoints (serve item, render FEN
SVG, record a decision) run in-process here using only stdlib + ``python-chess``.

It is loaded as an *external* plugin: add this repo's ``plugins/`` dir to the compute-service's
``~/.3lc-compute/settings.json`` ``plugin_dirs`` (see README.md). The package self-adds the repo
root to ``sys.path`` so its subprocess calls resolve ``-m scripts.curation.*``.

Requires in the compute-service venv: ``python-chess`` (``pip install chess``). Everything else
runs in the ChessVision ``.venv``.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import chess
import chess.svg
from litestar import Controller, Response, get, post
from litestar.params import FromPath

from tlc_compute.plugins.base import ComputePlugin, _ICON_SVG
from tlc_compute.plugins.registry import register
from tlc_compute.shared.async_job_store import AsyncJobStore

# ── Paths into the ChessVision repo (this file lives at <repo>/plugins/chessvision_testset/) ──
REPO_ROOT = Path(__file__).resolve().parents[2]
VENV_PY = REPO_ROOT / ".venv" / "bin" / "python"
TEST_ROOT = REPO_ROOT / "data" / "test"
LEDGER_PATH = REPO_ROOT / "data" / "provenance" / "ledger.json"

_store = AsyncJobStore(max_completed=40)


# ---------------------------------------------------------------------------
# Heavy stages — shell into the ChessVision .venv (reuses its torch/tlc/ultralytics)
# ---------------------------------------------------------------------------


def _shell(args: list[str], env_extra: dict[str, str], title: str) -> dict[str, Any]:
    """Run ``VENV_PY -m <module> ...`` in the repo, capturing output. For AsyncJobStore."""
    env = {**os.environ, **env_extra}
    proc = subprocess.run(  # noqa: S603 — fixed interpreter + module args, no shell
        [str(VENV_PY), *args],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-12:])
    return {"success": proc.returncode == 0, "title": title, "log": tail}


# Note: AsyncJobStore injects extra kwargs (e.g. a `job` progress handle) into the worker —
# absorb them with **_kwargs (same pattern as the built-in splitter plugin).
def _sample_job(batch: str, n: int, seed: int, **_kwargs: Any) -> dict[str, Any]:
    return _shell(
        ["-m", "scripts.curation.sample_test_candidates", "--batch", batch, "--n", str(n), "--seed", str(seed)],
        {},
        f"Sampling {n} candidates → {batch}",
    )


def _predict_job(batch: str, **_kwargs: Any) -> dict[str, Any]:
    return _shell(
        ["-m", "scripts.curation.predict_test_candidates", "--batch", batch],
        {"CHESSVISION_ALLOW_BARE_ULTRALYTICS": "1"},
        f"Predicting draft FENs → {batch}",
    )


def _commit_job(batch: str, **_kwargs: Any) -> dict[str, Any]:
    return _shell(
        ["-m", "scripts.curation.testset_tables", "--batch", batch, "--rebuild-tip"],
        {"TLC_DISABLE_ACCOUNT_SERVICE": "1"},
        f"Committing {batch} to 3LC tables + rebuilding tip",
    )


# ---------------------------------------------------------------------------
# Lightweight review helpers — in-process, stdlib + python-chess only
# ---------------------------------------------------------------------------


def _render_svg(board_fen: str) -> str:
    try:
        return chess.svg.board(board=chess.BaseBoard(board_fen), size=360)
    except (ValueError, IndexError):
        return f'<svg xmlns="http://www.w3.org/2000/svg" width="360" height="40"><text x="4" y="24" fill="red">invalid FEN</text></svg>'


def _load_json(path: Path, default: Any) -> Any:
    return json.loads(path.read_text()) if path.exists() else default


def _save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")


def _order(batch: str) -> list[str]:
    raw = TEST_ROOT / batch / "raw"
    return sorted(p.stem for p in raw.glob("*.JPG")) if raw.is_dir() else []


def _record_ledger(uuid: str, split: str, batch: str) -> None:
    ledger = _load_json(LEDGER_PATH, {})
    entry = ledger.get(uuid, {})
    entry.update({"split": split, "batch": batch})
    ledger[uuid] = entry
    _save_json(LEDGER_PATH, dict(sorted(ledger.items())))


# ---------------------------------------------------------------------------
# Litestar controller
# ---------------------------------------------------------------------------


class ChessVisionTestSetController(Controller):
    path = "/api/plugins/chessvision-testset"

    @get("/ui", media_type="text/html")
    async def ui(self) -> Response:
        from tlc_compute.plugins.registry import get_plugin

        plugin = get_plugin("chessvision-testset")
        return Response(content=plugin.get_ui_fragment() if plugin else "", media_type="text/html")

    @get("/batches")
    async def batches(self) -> dict[str, Any]:
        """List batches with progress counts so the UI can populate its selector."""
        out = []
        if TEST_ROOT.is_dir():
            for d in sorted(p for p in TEST_ROOT.iterdir() if p.is_dir()):
                raw = len(list((d / "raw").glob("*.JPG"))) if (d / "raw").is_dir() else 0
                gt = len(list((d / "ground_truth").glob("*.txt"))) if (d / "ground_truth").is_dir() else 0
                predicted = (d / "predictions.json").exists()
                out.append({"batch": d.name, "candidates": raw, "labeled": gt, "predicted": predicted})
        return {"batches": out}

    # ── heavy stages → async jobs ──────────────────────────────
    @post("/sample")
    async def sample(self, data: dict[str, Any]) -> dict[str, Any]:
        batch = (data.get("batch") or "").strip()
        if not batch:
            return {"error": "batch name required"}
        n, seed = int(data.get("n", 260)), int(data.get("seed", 42))
        return {"job_id": _store.submit(_sample_job, args=(batch, n, seed), title=f"Sampling {batch}")}

    @post("/predict")
    async def predict(self, data: dict[str, Any]) -> dict[str, Any]:
        batch = (data.get("batch") or "").strip()
        if not batch:
            return {"error": "batch name required"}
        return {"job_id": _store.submit(_predict_job, args=(batch,), title=f"Predicting {batch}")}

    @post("/commit")
    async def commit(self, data: dict[str, Any]) -> dict[str, Any]:
        batch = (data.get("batch") or "").strip()
        if not batch:
            return {"error": "batch name required"}
        return {"job_id": _store.submit(_commit_job, args=(batch,), title=f"Committing {batch}")}

    @get("/jobs/{job_id:str}")
    async def job(self, job_id: FromPath[str]) -> dict[str, Any]:
        job = _store.get(job_id)
        if not job:
            return {"error": "Job not found"}
        elapsed = (job["finished"] or time.time()) - job["started"]
        resp = {"id": job["id"], "status": job["status"], "elapsed": round(elapsed, 1)}
        if job["result"] is not None:
            resp["result"] = job["result"]
        return resp

    # ── review (in-process) ────────────────────────────────────
    @get("/item")
    async def item(self, batch: str, i: int | None = None) -> dict[str, Any]:
        order = _order(batch)
        if not order:
            return {"error": f"no candidates for batch {batch!r}"}
        review = _load_json(TEST_ROOT / batch / "review.json", {})
        preds = _load_json(TEST_ROOT / batch / "predictions.json", {})
        gt_dir = TEST_ROOT / batch / "ground_truth"
        committed = {p.stem for p in gt_dir.glob("*.txt")} if gt_dir.is_dir() else set()

        # An item is "done" if decided this session OR already has committed ground truth
        # (so pre-labeled batches like `initial` don't all read as unreviewed/failed).
        done = set(review) | committed
        reviewed = sum(1 for u in order if u in done)
        if i is None:
            i = next((k for k, u in enumerate(order) if u not in done), 0)
        i = max(0, min(i, len(order) - 1))

        uuid = order[i]
        pred, entry = preds.get(uuid, {}), review.get(uuid, {})
        fen_flipped = pred.get("fen_flipped")
        gt_fen = (gt_dir / f"{uuid}.txt").read_text().strip() if uuid in committed else None
        # Precedence: in-session edit/decision > committed ground truth > model draft.
        display_fen = entry.get("fen") or gt_fen or pred.get("fen")
        status = entry.get("status") or ("accepted" if gt_fen else "pending")
        return {
            "index": i, "uuid": uuid, "status": status,
            "total": len(order), "reviewed": reviewed, "remaining": len(order) - reviewed,
            "has_board": bool(display_fen),
            "fen": display_fen, "fen_flipped": fen_flipped,
            "svg_display": _render_svg(display_fen) if display_fen
            else "<i>no board / extraction failed — reject or hand-enter a FEN</i>",
            "svg_flipped": _render_svg(fen_flipped) if fen_flipped else "",
        }

    @get("/svg", media_type="text/html")
    async def svg(self, fen: str = "") -> str:
        return _render_svg(fen)

    @post("/decide")
    async def decide(self, data: dict[str, Any]) -> dict[str, Any]:
        batch, uuid, status, fen = data["batch"], data["uuid"], data["status"], data.get("fen")
        gt_file = TEST_ROOT / batch / "ground_truth" / f"{uuid}.txt"
        if status == "accepted" and fen:
            try:
                chess.BaseBoard(fen)
            except (ValueError, IndexError):
                return {"ok": False, "error": "invalid FEN"}
            gt_file.parent.mkdir(parents=True, exist_ok=True)
            gt_file.write_text(fen + "\n")
            _record_ledger(uuid, "test", batch)
        else:
            status = "rejected"
            gt_file.unlink(missing_ok=True)
            _record_ledger(uuid, "rejected", batch)
        review_path = TEST_ROOT / batch / "review.json"
        review = _load_json(review_path, {})
        review[uuid] = {"status": status, "fen": fen if status == "accepted" else None}
        _save_json(review_path, dict(sorted(review.items())))
        return {"ok": True}

    @get("/raw", media_type="image/jpeg")
    async def raw(self, batch: str, uuid: str) -> Response:
        p = TEST_ROOT / batch / "raw" / f"{uuid}.JPG"
        return Response(content=p.read_bytes(), media_type="image/jpeg") if p.exists() else Response(b"", status_code=404)

    @get("/board", media_type="image/png")
    async def board(self, batch: str, uuid: str) -> Response:
        p = TEST_ROOT / batch / "boards" / f"{uuid}.png"
        return Response(content=p.read_bytes(), media_type="image/png") if p.exists() else Response(b"", status_code=404)


# ---------------------------------------------------------------------------
# Plugin class
# ---------------------------------------------------------------------------


class ChessVisionTestSetPlugin(ComputePlugin):
    """Sidebar plugin: curate a ChessVision test-data batch and commit it to 3LC tables."""

    id = "chessvision-testset"
    name = "ChessVision Test Set"
    description = "Sample upload candidates, draft FENs with the model, review them, and commit a test batch to 3LC tables."
    version = "0.1.0"
    min_service_version = "0.1.0"
    icon = "♟"
    icon_svg = _ICON_SVG + '><rect x="2" y="2" width="12" height="12" rx="1.5"/><path d="M5 11h6M6 5l2-2 2 2-1 4H7z"/></svg>'
    display_mode = "sidebar"
    compatible_with = ["table"]
    input_types: list[str] = []  # standalone — sources its own raw data
    output_types = ["table"]
    section = "Data Ops"
    priority = 60
    quick_action = True
    quick_action_label = "Add Test Batch"
    quick_action_description = "Curate & commit a ChessVision test-data batch"
    # Image/SVG endpoints are loaded via <img src>/inline and carry no auth header.
    auth_exempt_paths = [
        r"^/api/plugins/chessvision-testset/raw$",
        r"^/api/plugins/chessvision-testset/board$",
        r"^/api/plugins/chessvision-testset/svg$",
    ]

    _ui_cache: str | None = None

    def get_ui_fragment(self) -> str:
        if self._ui_cache is None:
            self._ui_cache = (Path(__file__).resolve().parent / "ui.html").read_text(encoding="utf-8")
        return self._ui_cache

    def compute(self, params: dict[str, Any]) -> dict[str, Any]:
        return {"error": "Use the plugin UI / POST endpoints under /api/plugins/chessvision-testset."}

    def get_route_handlers(self) -> list[Any]:
        return [ChessVisionTestSetController]


register(ChessVisionTestSetPlugin())
