"""Sample a diverse batch of candidate boards from the raw uploads for a new test set.

Pure Python, no ``tlc`` — runs fully offline. The pipeline is deliberately simple and
auditable, because the whole point is *trustworthy* test data:

  1. Seed the provenance ledger from on-disk splits (so we never sample a training image).
  2. Index the uploads root (cached; ``--reindex`` to rebuild). Uploads are laid out as
     ``YYYY/MM/DD/<uuid>.JPG``.
  3. Allocate the sample budget across calendar months proportionally (largest-remainder),
     so the batch spreads across the full collection period rather than clustering in the
     heaviest months. This gives cheap temporal/usage diversity. (Visual-category diversity
     via embeddings is a deliberate later refinement — see ROADMAP.)
  4. Sample seeded-randomly within each month, excluding any UUID already in the ledger.
  5. Copy candidates into ``data/test/<batch>/raw/`` and record them as ``candidate`` in the
     ledger (reserving them so re-runs won't re-pick them). They become ``test`` only after a
     human confirms a FEN in the review tool.

Example:
    python -m scripts.curation.sample_test_candidates --batch harder-v1 --n 260 --seed 42
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import shutil
from pathlib import Path

from chessvision import constants
from scripts.curation import provenance
from scripts.utils import setup_logger

logger = setup_logger(__name__)

DEFAULT_UPLOADS_ROOT = Path("/media/gubbis/LACIE/chessvision-raw-uploads")
INDEX_PATH = constants.DATA_ROOT / "provenance" / "uploads_index.txt"


def build_index(uploads_root: Path) -> list[str]:
    """Walk ``YYYY/MM/DD`` and return relative ``YYYY/MM/DD/<uuid>.JPG`` paths."""
    logger.info(f"Indexing uploads under {uploads_root} (this can take a minute)...")
    rels: list[str] = []
    for year in sorted(p for p in uploads_root.iterdir() if p.is_dir() and p.name.isdigit()):
        for month in sorted(p for p in year.iterdir() if p.is_dir() and p.name.isdigit()):
            for day in sorted(p for p in month.iterdir() if p.is_dir() and p.name.isdigit()):
                for img in day.iterdir():
                    if img.suffix.upper() in (".JPG", ".JPEG", ".PNG"):
                        rels.append(f"{year.name}/{month.name}/{day.name}/{img.name}")
    logger.info(f"Indexed {len(rels):,} images.")
    return rels


def load_or_build_index(uploads_root: Path, reindex: bool) -> list[str]:
    if INDEX_PATH.exists() and not reindex:
        rels = INDEX_PATH.read_text().splitlines()
        logger.info(f"Loaded cached index ({len(rels):,} images) from {INDEX_PATH}.")
        return rels
    rels = build_index(uploads_root)
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.write_text("\n".join(rels) + "\n")
    logger.info(f"Cached index to {INDEX_PATH}.")
    return rels


def month_of(rel: str) -> str:
    """``2025/6/4/<uuid>.JPG`` -> ``2025/06`` (zero-padded for stable sorting)."""
    year, month, *_ = rel.split("/")
    return f"{year}/{int(month):02d}"


def uuid_of(rel: str) -> str:
    return Path(rel).stem


def allocate(month_sizes: dict[str, int], n: int) -> dict[str, int]:
    """Largest-remainder proportional allocation of ``n`` across months by eligible size."""
    total = sum(month_sizes.values())
    if total == 0:
        return {}
    n = min(n, total)
    exact = {m: n * size / total for m, size in month_sizes.items()}
    alloc = {m: int(v) for m, v in exact.items()}
    remainder = n - sum(alloc.values())
    # Hand out the remaining slots to the largest fractional parts.
    for m in sorted(exact, key=lambda m: exact[m] - alloc[m], reverse=True)[:remainder]:
        alloc[m] += 1
    return {m: c for m, c in alloc.items() if c > 0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", required=True, help="Batch name, e.g. 'harder-v1'.")
    parser.add_argument("--n", type=int, default=260, help="Number of candidates to sample.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility.")
    parser.add_argument("--uploads-root", type=Path, default=DEFAULT_UPLOADS_ROOT)
    parser.add_argument("--reindex", action="store_true", help="Rebuild the upload index cache.")
    parser.add_argument("--dry-run", action="store_true", help="Plan only; copy nothing.")
    args = parser.parse_args()

    if not args.uploads_root.is_dir():
        raise SystemExit(f"Uploads root not found: {args.uploads_root} (is the LaCie disk mounted?)")

    # 1. Ledger: seed from disk so training/test images are excluded.
    ledger = provenance.load_ledger()
    seeded = provenance.seed_from_disk(ledger)
    logger.info(f"Ledger: {len(ledger):,} known UUIDs ({seeded:,} newly seeded from disk).")
    excluded = provenance.used_uuids(ledger)

    # 2-3. Index, drop used UUIDs, group eligible images by month.
    rels = load_or_build_index(args.uploads_root, args.reindex)
    by_month: dict[str, list[str]] = collections.defaultdict(list)
    n_excluded = 0
    for rel in rels:
        if uuid_of(rel) in excluded:
            n_excluded += 1
            continue
        by_month[month_of(rel)].append(rel)
    logger.info(
        f"Eligible: {sum(len(v) for v in by_month.values()):,} images across "
        f"{len(by_month)} months ({n_excluded:,} excluded as already-used).",
    )

    # 4. Allocate and sample, seeded per-month for reproducibility.
    alloc = allocate({m: len(v) for m, v in by_month.items()}, args.n)
    rng = random.Random(args.seed)  # noqa: S311 — reproducible sampling, not cryptographic
    picked: list[str] = []
    logger.info("Per-month allocation:")
    for month in sorted(alloc):
        pool = sorted(by_month[month])  # sort first so sampling is deterministic
        chosen = rng.sample(pool, alloc[month])
        picked.extend(chosen)
        logger.info(f"  {month}: {alloc[month]:>3} / {len(pool):,} eligible")
    logger.info(f"Total sampled: {len(picked)}")

    if args.dry_run:
        logger.info("Dry run — no files copied, ledger not modified.")
        return

    # 5. Copy into the batch's raw/ dir and record as candidates.
    batch_raw = constants.DATA_ROOT / "test" / args.batch / "raw"
    batch_raw.mkdir(parents=True, exist_ok=True)
    manifest = []
    for rel in picked:
        uuid = uuid_of(rel)
        dst = batch_raw / f"{uuid}.JPG"
        shutil.copy2(args.uploads_root / rel, dst)
        provenance.record(ledger, uuid, provenance.CANDIDATE, batch=args.batch, source=rel)
        manifest.append({"uuid": uuid, "source": rel, "month": month_of(rel)})

    manifest_path = constants.DATA_ROOT / "test" / args.batch / "candidates_manifest.json"
    manifest_path.write_text(json.dumps({"batch": args.batch, "seed": args.seed, "items": manifest}, indent=2))
    provenance.save_ledger(ledger)
    logger.info(f"Copied {len(picked)} candidates to {batch_raw}")
    logger.info(f"Wrote manifest {manifest_path} and updated ledger ({len(ledger):,} UUIDs).")


if __name__ == "__main__":
    main()
