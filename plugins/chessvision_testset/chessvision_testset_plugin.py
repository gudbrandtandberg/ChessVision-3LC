"""ChessVision Test Set — a 3LC Insights (compute-service) **venv** plugin.

Wraps the "add a curated test-data batch" workflow as a hub sidebar plugin:

    sample candidates → model-assisted draft FENs → human review → commit to 3LC tables

Design: this plugin is a **thin orchestrator**. The heavy stages (sampling the LaCie upload
pool, running the YOLO pipeline, writing 3LC tables) are run by shelling into the ChessVision
repo's own ``.venv`` — so the *plugin's* venv stays light and we reuse ChessVision's exact
dependencies (torch/ultralytics/tlc). The lightweight review endpoints (serve item, render FEN
SVG, record a decision) run in-process here using only stdlib + ``python-chess``.

Isolation: this is a ``venv`` plugin (``isolation = "venv"`` in ``plugin.toml``). The host never
imports it; a worker process runs it out-of-process in the plugin's own uv venv (``pyproject.toml``
next to ``plugin.toml`` declares ``3lc-compute`` + ``chess``). That keeps ``python-chess`` out of
the main compute-service venv. The same ``run_job``/route code runs identically whether served
in-process (host tier) or proxied to the worker (venv tier).

Layout: the worker launches with the plugin dir as ``cwd`` and adds it to ``sys.path``, so the
entrypoint must be a **flat module** living directly in the plugin dir — hence this file (rather
than a ``chessvision_testset`` package, which would need the parent dir on the path). The module
stays at the same depth as the old ``__init__.py``, so ``REPO_ROOT`` (``parents[2]``) is unchanged.

Contract: **behavior-only** on the current compute-service SDK. All metadata lives in
``plugin.toml`` (the manifest); there is no ``register()`` call and no metadata on the class. The
three heavy stages run through the unified ``run_job(ctx)`` job contract (host-managed queue /
cancel / generic progress), dispatched by a ``stage`` param. Review endpoints are **relative**
Litestar route handlers returned by ``get_route_handlers()`` and served under
``/api/plugins/chessvision-testset/`` (no absolute-path controller, so nothing shadows ``/run``).

It is loaded as an *external* plugin: add this repo's ``plugins/`` dir to the compute-service's
``~/.3lc-compute/settings.json`` ``plugin_dirs`` (see README.md). Subprocess calls run with
``cwd`` set to the repo root, so ``-m scripts.curation.*`` resolves there.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

import chess
import chess.svg
from litestar import Response, get, post
from tlc_compute.plugin_sdk import ComputePlugin

if TYPE_CHECKING:
    from litestar.handlers import BaseRouteHandler
    from tlc_compute.plugin_sdk import JobContext

# ── Paths into the ChessVision repo (this module lives at <repo>/plugins/chessvision_testset/) ──
REPO_ROOT = Path(__file__).resolve().parents[2]
VENV_PY = REPO_ROOT / ".venv" / "bin" / "python"
TEST_ROOT = REPO_ROOT / "data" / "test"
LEDGER_PATH = REPO_ROOT / "data" / "provenance" / "ledger.json"

# Heavy stages, keyed by the `stage` job param. Each entry builds the subprocess
# module invocation, any extra env, and a human title.
_STAGES = ("sample", "predict", "commit")


# ---------------------------------------------------------------------------
# Heavy stages — shell into the ChessVision .venv (reuses its torch/tlc/ultralytics)
# ---------------------------------------------------------------------------


def _shell(args: list[str], env_extra: dict[str, str], title: str) -> dict[str, Any]:
    """Run ``VENV_PY -m <module> ...`` in the repo, capturing the tail of its output."""
    env = {**os.environ, **env_extra}
    proc = subprocess.run(  # noqa: S603 — fixed interpreter + module args, no shell
        [str(VENV_PY), *args],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-12:])
    return {"success": proc.returncode == 0, "title": title, "log": tail}


def _stage_command(stage: str, params: dict[str, Any], batch: str) -> tuple[list[str], dict[str, str], str]:
    """Build the (args, env_extra, title) for one heavy stage."""
    if stage == "sample":
        n, seed = int(params.get("n", 260)), int(params.get("seed", 42))
        args = ["-m", "scripts.curation.sample_test_candidates", "--batch", batch, "--n", str(n), "--seed", str(seed)]
        return args, {}, f"Sampling {n} candidates → {batch}"
    if stage == "predict":
        args = ["-m", "scripts.curation.predict_test_candidates", "--batch", batch]
        return args, {"CHESSVISION_ALLOW_BARE_ULTRALYTICS": "1"}, f"Predicting draft FENs → {batch}"
    # commit
    args = ["-m", "scripts.curation.testset_tables", "--batch", batch, "--rebuild-tip"]
    return args, {"TLC_DISABLE_ACCOUNT_SERVICE": "1"}, f"Committing {batch} to 3LC tables + rebuilding tip"


# ---------------------------------------------------------------------------
# Lightweight review helpers — in-process, stdlib + python-chess only
# ---------------------------------------------------------------------------


def _render_svg(board_fen: str) -> str:
    try:
        return chess.svg.board(board=chess.BaseBoard(board_fen), size=360)
    except (ValueError, IndexError):
        return '<svg xmlns="http://www.w3.org/2000/svg" width="360" height="40"><text x="4" y="24" fill="red">invalid FEN</text></svg>'


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
# Plugin class — behavior only (metadata lives in plugin.toml)
# ---------------------------------------------------------------------------


class ChessVisionTestSetPlugin(ComputePlugin):
    """Sidebar plugin: curate a ChessVision test-data batch and commit it to 3LC tables.

    Behavior only — the host hydrates ``id``/``name``/``icon``/``version`` onto the
    instance from the manifest after construction.
    """

    # Display identity stamped onto the instance by the host from the manifest.
    id: str

    _ui_cache: str | None = None

    def get_ui_fragment(self) -> str:
        """Return the self-contained review/curation UI fragment.

        Injects ``window.PluginJobs`` so the UI drives the heavy stages over the
        generic ``job_update`` channel and receives the structured ``stage_result``
        event (the flat generic schema can't carry the subprocess log tail).
        """
        if self._ui_cache is None:
            from tlc_compute.plugin_sdk.shared.job_tracker import job_tracker_script
            from tlc_compute.plugin_sdk.shared.ui_inject import inject_scripts

            raw = (Path(__file__).resolve().parent / "ui.html").read_text(encoding="utf-8")
            self._ui_cache = inject_scripts(raw, job_tracker_script())
        return self._ui_cache

    def compute(self, params: dict[str, Any]) -> dict[str, Any]:
        """Not used — heavy stages run via ``POST /api/plugins/chessvision-testset/run``."""
        return {"error": "Use the plugin UI / POST /api/plugins/chessvision-testset/run (stage=sample|predict|commit)."}

    def run_job(self, ctx: JobContext) -> None:
        """Run one heavy stage (sample / predict / commit) as a job.

        The stage is selected by ``ctx.params['stage']`` and shells into the
        ChessVision ``.venv``. Generic progress (percent + label) feeds the Queue &
        Progress panel; the subprocess outcome + log tail ride the plugin-specific
        ``stage_result`` event for the embedded UI's log pane. A non-zero subprocess
        exit is a *reported outcome* (``success: false`` on ``stage_result``), not a
        job crash, so the job still completes — mirroring the old ``/jobs`` result.

        Args:
            ctx: Host-provided job context. ``ctx.params`` carries ``stage`` and
                ``batch`` (plus ``n``/``seed`` for the sample stage).

        Raises:
            ValueError: If ``stage`` is unknown or ``batch`` is missing.

        """
        stage = str(ctx.params.get("stage", "")).strip()
        if stage not in _STAGES:
            msg = f"unknown stage: {stage!r} (expected one of {_STAGES})"
            raise ValueError(msg)
        batch = str(ctx.params.get("batch", "")).strip()
        if not batch:
            msg = "batch name required"
            raise ValueError(msg)

        args, env_extra, title = _stage_command(stage, ctx.params, batch)
        ctx.progress(percent=5, label=title)
        result = _shell(args, env_extra, title)
        ctx.progress(percent=100, label=title + (" ✓" if result["success"] else " ✗"))
        ctx.emit("stage_result", {"stage": stage, **result})

    def get_route_handlers(self) -> list[BaseRouteHandler]:
        """Return the review/curation routes as relative Litestar handlers.

        Served under ``/api/plugins/chessvision-testset/`` — no static controller
        node, so nothing shadows the generic ``/run`` route. In venv mode the worker
        mounts these on its own Litestar app (litestar is a base venv dependency).
        Handlers are ``def`` (Litestar runs them in a threadpool via
        ``sync_to_thread``) because they touch the filesystem.
        """

        @get("/batches", sync_to_thread=True)
        def batches() -> dict[str, Any]:
            """List batches with progress counts so the UI can populate its selector."""
            out = []
            if TEST_ROOT.is_dir():
                for d in sorted(p for p in TEST_ROOT.iterdir() if p.is_dir()):
                    raw = len(list((d / "raw").glob("*.JPG"))) if (d / "raw").is_dir() else 0
                    gt = len(list((d / "ground_truth").glob("*.txt"))) if (d / "ground_truth").is_dir() else 0
                    predicted = (d / "predictions.json").exists()
                    out.append({"batch": d.name, "candidates": raw, "labeled": gt, "predicted": predicted})
            return {"batches": out}

        @get("/item", sync_to_thread=True)
        def item(batch: str, i: int | None = None) -> dict[str, Any]:
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

        @get("/svg", media_type="text/html", sync_to_thread=True)
        def svg(fen: str = "") -> str:
            return _render_svg(fen)

        @post("/decide", status_code=200, sync_to_thread=True)
        def decide(data: dict[str, Any]) -> dict[str, Any]:
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

        @get("/raw", media_type="image/jpeg", sync_to_thread=True)
        def raw(batch: str, uuid: str) -> Response[bytes]:
            p = TEST_ROOT / batch / "raw" / f"{uuid}.JPG"
            return Response(content=p.read_bytes(), media_type="image/jpeg") if p.exists() else Response(b"", status_code=404)

        @get("/board", media_type="image/png", sync_to_thread=True)
        def board(batch: str, uuid: str) -> Response[bytes]:
            p = TEST_ROOT / batch / "boards" / f"{uuid}.png"
            return Response(content=p.read_bytes(), media_type="image/png") if p.exists() else Response(b"", status_code=404)

        return [batches, item, svg, decide, raw, board]
