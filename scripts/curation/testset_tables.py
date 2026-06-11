"""Materialize curated test batches as 3LC tables, with a joined "tip" over all batches.

This is the data-management layer the review workflow commits into. The on-disk files stay
the human-facing source of truth (raw photos under ``data/test/<batch>/raw`` + confirmed FENs
under ``ground_truth/<uuid>.txt``, plus the provenance ledger); this module *derives* 3LC
tables from them, so tables are always reproducible from the repo.

Model (reusing the existing `chessvision-testing` / `test` project+dataset):
  - **Per-batch table** — one table per batch, named by the batch slug (e.g. ``harder-v1``,
    ``initial``). Columns: ``image`` (url ref to the raw photo), ``fen`` (ground-truth board
    FEN), ``source`` (upload provenance path), ``month`` (sampling stratum). Only *labeled*
    rows (those with a ground_truth FEN) become rows — rejected/unreviewed candidates don't.
  - **Joined tip** — ``test-all``, rebuilt from scratch as ``join_tables`` of every labeled
    batch. The tip is derived/auditable; batches are the source of truth. ``.latest()`` on the
    tip is the current aggregate the eval should run against.

Versioning: batch slug + a stable tip name; 3LC revisions/lineage handle the rest. Re-committing
a batch overwrites its table; rebuilding the tip overwrites ``test-all`` and records the batch
tables as its lineage inputs.

Pure ``tlc`` (no model). Run offline with ``TLC_DISABLE_ACCOUNT_SERVICE=1``:
    TLC_DISABLE_ACCOUNT_SERVICE=1 .venv/bin/python -m scripts.curation.testset_tables --rebuild-tip
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import tlc

from chessvision import constants
from scripts.utils import setup_logger

logger = setup_logger(__name__)

PROJECT = "chessvision-testing"
DATASET = "test"
TIP_TABLE = "test-all"
TEST_ROOT = constants.DATA_ROOT / "test"


def _manifest_lookup(batch_dir: Path) -> dict[str, dict]:
    """Map uuid -> {source, month} from a batch's candidates_manifest.json, if present.

    Older batches (e.g. ``initial``) predate the manifest; their provenance is simply unknown.
    """
    manifest = batch_dir / "candidates_manifest.json"
    if not manifest.exists():
        return {}
    items = json.loads(manifest.read_text()).get("items", [])
    return {it["uuid"]: it for it in items}


def _read_batch_rows(batch_dir: Path) -> dict[str, list]:
    """Collect labeled rows for a batch as parallel column lists.

    A row is included only when both a raw image and a confirmed ground-truth FEN exist.
    """
    gt_dir = batch_dir / "ground_truth"
    raw_dir = batch_dir / "raw"
    lookup = _manifest_lookup(batch_dir)

    images, fens, sources, months = [], [], [], []
    for gt_file in sorted(gt_dir.glob("*.txt")) if gt_dir.is_dir() else []:
        uuid = gt_file.stem
        img = raw_dir / f"{uuid}.JPG"
        if not img.exists():
            logger.warning(f"  {uuid}: ground truth without raw image, skipping")
            continue
        fen = gt_file.read_text().strip()
        if not fen:
            continue
        images.append(tlc.Url(str(img.resolve())).to_str())
        fens.append(fen)
        meta = lookup.get(uuid, {})
        sources.append(meta.get("source", ""))
        months.append(meta.get("month", ""))
    return {"image": images, "fen": fens, "source": sources, "month": months}


def build_batch_table(batch: str, *, project: str = PROJECT, if_exists: str = "overwrite") -> tlc.Table | None:
    """Build (or overwrite) the per-batch table from its labeled files. None if no labeled rows."""
    data = _read_batch_rows(TEST_ROOT / batch)
    n = len(data["fen"])
    if n == 0:
        logger.info(f"Batch {batch!r}: no labeled rows yet — skipping.")
        return None

    schema = {
        "image": tlc.schemas.ImageSchema(sample_type="url"),
        "fen": tlc.schemas.StringSchema(string_role="fen"),
        "source": tlc.schemas.StringSchema(),
        "month": tlc.schemas.StringSchema(),
    }
    table = tlc.Table.from_dict(
        data,
        schema=schema,
        project_name=project,
        dataset_name=DATASET,
        table_name=batch,
        if_exists=if_exists,
        description=f"ChessVision test batch '{batch}' ({n} labeled boards).",
    )
    logger.info(f"Batch {batch!r}: {n} rows -> {table.url}")
    return table


def list_labeled_batches() -> list[str]:
    """Batch slugs under data/test/ that have at least one ground-truth FEN."""
    out = []
    for batch_dir in sorted(p for p in TEST_ROOT.iterdir() if p.is_dir()):
        gt = batch_dir / "ground_truth"
        if gt.is_dir() and any(gt.glob("*.txt")):
            out.append(batch_dir.name)
    return out


def rebuild_tip(*, project: str = PROJECT, tip_name: str = TIP_TABLE) -> tlc.Table | None:
    """Rebuild every labeled batch table, then join them into the tip. None if nothing labeled."""
    batches = list_labeled_batches()
    tables = [t for b in batches if (t := build_batch_table(b, project=project)) is not None]
    if not tables:
        logger.info("No labeled batches found — nothing to join.")
        return None
    if len(tables) == 1:
        logger.info(f"Only one labeled batch; tip == that batch ({tables[0].url}).")
    tip = tlc.Table.join_tables(
        tables,
        project_name=project,
        dataset_name=DATASET,
        table_name=tip_name,
        if_exists="overwrite",
    )
    total = sum(len(t) for t in tables)
    logger.info(f"Tip {tip_name!r}: {len(tables)} batches, {total} rows -> {tip.url}")
    return tip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", help="Build just this batch's table.")
    parser.add_argument("--rebuild-tip", action="store_true", help="Build all labeled batches + the joined tip.")
    parser.add_argument("--project", default=PROJECT, help="Override project name (e.g. for a dry run).")
    parser.add_argument("--tip-name", default=TIP_TABLE)
    args = parser.parse_args()

    if not args.batch and not args.rebuild_tip:
        parser.error("pass --batch <slug> and/or --rebuild-tip")
    if args.batch:
        build_batch_table(args.batch, project=args.project)
    if args.rebuild_tip:
        rebuild_tip(project=args.project, tip_name=args.tip_name)


if __name__ == "__main__":
    main()
