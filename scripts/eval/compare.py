#!/usr/bin/env python
"""Compare current ChessVision eval metrics against a committed baseline.

Runs the evaluation suite (aggregate metrics only, no per-square metrics table) and diffs the
key metrics against ``benchmarks/baseline.json``. This is what lets a change be *measured*
rather than guessed at.

Usage::

    # Print current-vs-baseline (exits non-zero if accuracy regresses beyond --tolerance)
    python scripts/eval/compare.py

    # (Re)write the baseline from the current run — do this after retraining the
    # "current best" models, since weights are not committed.
    python scripts/eval/compare.py --update

Note: requires the local 3LC source setup (see CLAUDE.md) — runs with a throwaway key.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from chessvision import constants
from scripts.eval.evaluate import evaluate_model

REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_PATH = REPO_ROOT / "benchmarks" / "baseline.json"

# Higher-is-better metrics we guard against regressions.
ACCURACY_METRICS = [
    "top_1_accuracy",
    "top_1_accuracy_validated",
    "top_2_accuracy",
    "top_3_accuracy",
]
# Shown for context; not gated (lower-is-better or informational).
CONTEXT_METRICS = [
    "extraction_failures",
    "validation_fixes",
    "validation_improvements",
    "avg_time_per_prediction",
]


def run_eval(table_name: str, project_name: str, image_folder: Path) -> dict:
    """Run the aggregate evaluation and return the test_results dict."""
    run = evaluate_model(
        image_folder=image_folder,
        table_name=table_name,
        project_name=project_name,
        run_name="compare",
        board_extractor_model_id="yolo",
        classifier_model_id="yolo",
        include_metrics_table=False,
    )
    return dict(run.constants["parameters"]["test_results"])


def load_baseline() -> dict | None:
    if not BASELINE_PATH.exists():
        return None
    with BASELINE_PATH.open() as f:
        return json.load(f)


def write_baseline(results: dict, table_name: str) -> None:
    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Store weight filenames, not absolute machine-specific paths, since this file is committed.
    metrics = dict(results)
    for key in ("board_extractor_weights", "classifier_weights"):
        if isinstance(metrics.get(key), str):
            metrics[key] = Path(metrics[key]).name
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "table_name": table_name,
        "metrics": metrics,
    }
    with BASELINE_PATH.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote baseline to {BASELINE_PATH}")


def _fmt(value: float | int | str) -> str:
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def print_comparison(current: dict, baseline: dict, tolerance: float) -> bool:
    """Print a metric table. Returns True if any accuracy metric regressed."""
    base_metrics = baseline["metrics"]
    print(f"\nBaseline created {baseline.get('created_utc', '?')} on table '{baseline.get('table_name', '?')}'")
    print(f"{'metric':<28}{'baseline':>12}{'current':>12}{'delta':>12}")
    print("-" * 64)

    regressed = False
    for key in [*ACCURACY_METRICS, *CONTEXT_METRICS]:
        if key not in current:
            continue
        cur = current[key]
        base = base_metrics.get(key)
        if isinstance(cur, int | float) and isinstance(base, int | float):
            delta = cur - base
            flag = ""
            if key in ACCURACY_METRICS and delta < -tolerance:
                flag = "  <-- REGRESSION"
                regressed = True
            print(f"{key:<28}{_fmt(base):>12}{_fmt(cur):>12}{delta:>+12.4f}{flag}")
        else:
            print(f"{key:<28}{_fmt(base):>12}{_fmt(cur):>12}{'':>12}")
    print("-" * 64)
    return regressed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table-name", default="initial")
    parser.add_argument("--project-name", default="chessvision-testing")
    parser.add_argument(
        "--image-folder",
        type=Path,
        default=constants.DATA_ROOT / "test" / "initial" / "raw",
        help="Used only if the table does not already exist.",
    )
    parser.add_argument("--tolerance", type=float, default=0.005, help="Allowed accuracy drop before failing.")
    parser.add_argument("--update", action="store_true", help="Write current results as the new baseline.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results = run_eval(args.table_name, args.project_name, args.image_folder)

    if args.update:
        write_baseline(results, args.table_name)
        return 0

    baseline = load_baseline()
    if baseline is None:
        print(f"No baseline at {BASELINE_PATH}. Run with --update to create one.")
        for key in [*ACCURACY_METRICS, *CONTEXT_METRICS]:
            if key in results:
                print(f"  {key:<28}{_fmt(results[key]):>12}")
        return 0

    regressed = print_comparison(results, baseline, args.tolerance)
    if regressed:
        print("FAIL: at least one accuracy metric regressed beyond tolerance.")
        return 1
    print("OK: no accuracy regressions.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
