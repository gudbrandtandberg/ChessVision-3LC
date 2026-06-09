"""Create a class-balanced revision of the piece classification training table.

For each row, sets the `weight` column to `(N_total / num_classes) / count(class)` so that
each class contributes equally to the expected sampled batch while the average weight stays
at 1.0 (so the dataloader length is unchanged with `tlc.create_sampler`).
"""

from __future__ import annotations

import collections
import logging

import tlc
from tlc._core.objects.tables.from_table.edited_table import EditedTable

from chessvision import constants
from scripts.train import config
from scripts.utils import setup_logger

NEW_TABLE_NAME = "class-balanced"


def build_balanced_weights(table: tlc.Table) -> tuple[dict[int, float], list[list[int] | float]]:
    rows = list(table.table_rows)
    counts = collections.Counter(row["label"] for row in rows)
    n_total = len(rows)
    n_classes = len(counts)
    avg_count = n_total / n_classes

    class_weights = {label: avg_count / count for label, count in counts.items()}

    indices_per_class: dict[int, list[int]] = {label: [] for label in counts}
    for idx, row in enumerate(rows):
        indices_per_class[row["label"]].append(idx)

    runs_and_values: list[list[int] | float] = []
    for label, indices in indices_per_class.items():
        runs_and_values.append(indices)
        runs_and_values.append(float(class_weights[label]))

    return class_weights, runs_and_values


def main() -> None:
    logger = setup_logger(__name__)

    source = tlc.Table.from_names(
        table_name=config.INITIAL_TABLE_NAME,
        dataset_name=config.PIECE_CLASSIFICATION_DATASETS["train"],
        project_name=config.PIECE_CLASSIFICATION_PROJECT,
    )
    logger.info(f"Source table: {source.url} ({len(source)} rows)")

    class_weights, runs_and_values = build_balanced_weights(source)
    counts = collections.Counter(row["label"] for row in source.table_rows)
    logger.info("Per-class weights (label idx | name | count | weight):")
    for label_idx in sorted(class_weights):
        logger.info(
            f"  {label_idx:>2} {constants.LABEL_NAMES[label_idx]:>2}  "
            f"count={counts[label_idx]:>5}  weight={class_weights[label_idx]:.4f}",
        )

    target_url = source.url.create_sibling(NEW_TABLE_NAME)
    try:
        existing = tlc.Table.from_url(target_url)
        logger.warning(f"Table {NEW_TABLE_NAME!r} already exists at {existing.url}; re-using it.")
        balanced = existing
    except (FileNotFoundError, ValueError):
        balanced = EditedTable(
            input_table_url=source,
            edits={"weight": {"runs_and_values": runs_and_values}},
            url=target_url,
        )
        balanced.ensure_fully_defined()
        logger.info(f"Wrote balanced table: {balanced.url}")

    sample_weights = [row["weight"] for row in balanced.table_rows]
    logger.info(
        f"Sanity check: mean weight={sum(sample_weights) / len(sample_weights):.4f}, "
        f"min={min(sample_weights):.4f}, max={max(sample_weights):.4f}",
    )


if __name__ == "__main__":
    main()
