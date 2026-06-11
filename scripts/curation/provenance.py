"""Provenance ledger: a single source of truth for which raw upload UUID belongs to
which data split.

Motivation: the raw uploads (~695k images on the LaCie disk) are an effectively infinite
pool. To keep splits honest we must never let an image used for training leak into the test
set (or vice-versa). Every image is UUID-named, both in the uploads and in the on-disk
splits, so a UUID is a stable global key.

The ledger maps ``uuid -> {split, batch, source, ts}`` where:
  - ``split``   one of: ``train_val`` (board-extraction training), ``test`` (confirmed test
                ground truth), ``candidate`` (sampled, awaiting human review), ``rejected``
                (reviewed and discarded, e.g. not a real board).
  - ``batch``   the named batch a candidate/test image was sampled into (or None).
  - ``source``  the relative path inside the uploads root it came from (or None for
                pre-existing on-disk data).
  - ``ts``      ISO timestamp it was recorded.

Pure Python — no ``tlc``. Safe to import and run fully offline.

Note on the classifier: the square-classification crops (``data/squares/``) carry their own
per-square UUIDs, not their source board's UUID, so they cannot be excluded by filename.
The extraction training set (``data/board_extraction/images``, 631 boards) is the practical
proxy for "boards the models have seen"; against a 695k pool, accidental reuse is ~0.1%.
"""

from __future__ import annotations

import datetime
import json

from chessvision import constants

LEDGER_PATH = constants.DATA_ROOT / "provenance" / "ledger.json"

# Valid split labels.
TRAIN_VAL = "train_val"
TEST = "test"
CANDIDATE = "candidate"
REJECTED = "rejected"


def load_ledger() -> dict[str, dict]:
    """Load the ledger, returning an empty dict if it does not exist yet."""
    if LEDGER_PATH.exists():
        with LEDGER_PATH.open() as f:
            return json.load(f)
    return {}


def save_ledger(ledger: dict[str, dict]) -> None:
    """Persist the ledger (sorted by UUID for stable diffs)."""
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER_PATH.open("w") as f:
        json.dump(dict(sorted(ledger.items())), f, indent=2)
        f.write("\n")


def record(
    ledger: dict[str, dict],
    uuid: str,
    split: str,
    *,
    batch: str | None = None,
    source: str | None = None,
    overwrite: bool = True,
) -> bool:
    """Record (or update) one UUID's split assignment.

    Returns True if the ledger was modified. When ``overwrite`` is False, an existing entry
    is left untouched (used for idempotent seeding from disk).
    """
    if not overwrite and uuid in ledger:
        return False
    ledger[uuid] = {
        "split": split,
        "batch": batch,
        "source": source,
        "ts": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    return True


def used_uuids(ledger: dict[str, dict]) -> set[str]:
    """All UUIDs that must be excluded from fresh sampling.

    Everything already in the ledger is off-limits, including ``rejected`` ones (no point
    re-surfacing an image a human already discarded) — the pool is large enough that we
    never need to reclaim them.
    """
    return set(ledger)


def seed_from_disk(ledger: dict[str, dict]) -> int:
    """Populate the ledger from images already committed on disk. Idempotent: existing
    entries are never overwritten, so a hand-edited split is preserved.

    Returns the number of newly added entries.
    """
    added = 0

    # Board-extraction training images -> train_val.
    extraction_dir = constants.DATA_ROOT / "board_extraction" / "images"
    if extraction_dir.is_dir():
        for img in extraction_dir.glob("*.JPG"):
            added += record(ledger, img.stem, TRAIN_VAL, overwrite=False)

    # Existing test sets -> test.
    test_root = constants.DATA_ROOT / "test"
    if test_root.is_dir():
        for raw_img in test_root.glob("*/raw/*.JPG"):
            added += record(ledger, raw_img.stem, TEST, batch=raw_img.parts[-3], overwrite=False)

    return added
