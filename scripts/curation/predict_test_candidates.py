"""Run the ChessVision pipeline over a batch of sampled candidates to produce *draft* FENs
for human review. This is the "model-assisted" half of building the test set.

For each candidate raw image we:
  - extract the board (YOLO-seg), and
  - if extraction succeeds, classify the position in BOTH orientations (flip=False/True),
    since a photo taken from black's side yields a 180°-rotated board. Storing both lets the
    reviewer pick the correct orientation with one keystroke instead of hand-editing.

Output: ``data/test/<batch>/predictions.json`` mapping uuid -> draft FENs + extraction status,
and grayscale extracted-board crops under ``data/test/<batch>/boards/`` for sanity-checking
extraction. Nothing here is ground truth — the review tool turns drafts into confirmed FENs.

No ``tlc`` is required for inference; run with the env that loads the YOLO weights, e.g.:
    set -a; source .env; set +a
    .venv/bin/python -m scripts.curation.predict_test_candidates --batch harder-v1
"""

from __future__ import annotations

import argparse
import json

import cv2

from chessvision import constants
from chessvision.core import ChessVision
from scripts.utils import setup_logger

logger = setup_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", required=True, help="Batch name, e.g. 'harder-v1'.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Board-extraction threshold.")
    args = parser.parse_args()

    batch_dir = constants.DATA_ROOT / "test" / args.batch
    raw_dir = batch_dir / "raw"
    if not raw_dir.is_dir():
        raise SystemExit(f"No raw/ dir for batch {args.batch!r}: {raw_dir}")

    boards_dir = batch_dir / "boards"
    boards_dir.mkdir(parents=True, exist_ok=True)

    cv = ChessVision()
    raw_images = sorted(raw_dir.glob("*.JPG"))
    logger.info(f"Predicting {len(raw_images)} candidates in batch {args.batch!r}...")

    predictions: dict[str, dict] = {}
    n_extracted = 0
    for i, img_path in enumerate(raw_images, 1):
        uuid = img_path.stem
        image = cv2.imread(str(img_path))
        if image is None:
            logger.warning(f"  [{i}/{len(raw_images)}] {uuid}: unreadable, skipping")
            predictions[uuid] = {"extraction_ok": False, "error": "unreadable"}
            continue

        board_result = cv.extract_board(image, args.threshold)
        if board_result.board_image is None:
            predictions[uuid] = {"extraction_ok": False}
            continue

        n_extracted += 1
        cv2.imwrite(str(boards_dir / f"{uuid}.png"), board_result.board_image)
        normal = cv.classify_position(board_result.board_image, flip=False)
        flipped = cv.classify_position(board_result.board_image, flip=True)
        predictions[uuid] = {
            "extraction_ok": True,
            "fen": normal.fen,
            "fen_flipped": flipped.fen,
            "original_fen": normal.original_fen,
        }
        if i % 25 == 0:
            logger.info(f"  [{i}/{len(raw_images)}] {n_extracted} boards extracted so far")

    out_path = batch_dir / "predictions.json"
    out_path.write_text(json.dumps(predictions, indent=2) + "\n")
    n_failed = len(raw_images) - n_extracted
    logger.info(
        f"Done. Extracted {n_extracted}/{len(raw_images)} boards "
        f"({n_failed} extraction failures). Wrote {out_path}.",
    )


if __name__ == "__main__":
    main()
