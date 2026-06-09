# ChessVision Roadmap & Planning

> Living sync doc between Gudbrand and Claude. Claude owns the code; Gudbrand sets
> direction. Keep this short and current — prune done items, promote ideas as they ripen.

Last updated: 2026-06-08 (end of day 1 of the takeover)

## The two goals

1. **Do well at the end-to-end task**: image → FEN. Board segmentation → extraction →
   square classification → position validation. Push accuracy, robustness (harder test
   set, better extraction), and latency.
2. **Be a great 3LC demo/dev playground**: every stage should showcase a real 3LC
   workflow (data selection, model-assisted labeling, metrics analysis, lineage, sample
   weighting, coreset selection, embeddings). The ~600k user uploads are raw fuel for this.

When the two diverge, Claude trades off case-by-case and flags it here.

## Current state

- **Pipeline**: YOLO-seg board extraction → OpenCV contour/perspective transform → YOLO-cls
  piece classification → chess-rule validation (only "no pawns on rank 1/8" active; king/bishop
  count rules commented out). UNet/timm remain as fallbacks only.
- **Models shipped in-repo** (`weights/best_yolo_{classifier,extractor}.pt`), retrained on the
  GPU node and checked in under AGPL.
- **Baseline** (`benchmarks/baseline.json`, CUDA, initial 24-img test set): top-1 **0.986** /
  validated **0.988** / 0 extraction failures. Reproduce/diff with `scripts/eval/compare.py`.
- **License**: whole repo is **AGPL-3.0** (built on Ultralytics YOLO; weights inherit AGPL).
- **Dev**: runs fully offline-ish via editable `tlc`/`3lc-ultralytics` source + a throwaway key
  (`TLC_DISABLE_ACCOUNT_SERVICE=1`). See CLAUDE.md (Mac + godfire setup).
- **Branch/PR**: everything is on `claude/foundation-yolo-default-baseline` → PR #18 (open,
  not yet merged). GPU node `gubbis@godfire` is set up with a standalone `.venv`.

## Done (day 1)

- YOLO is the default board extractor; fixed two real bugs it exposed (non-contiguous `.view()`
  crash, half→float32 dtype) in `extract_board`.
- YOLO loaders prefer `3lc-ultralytics`, fail loud otherwise; bare ultralytics opt-in via
  `CHESSVISION_ALLOW_BARE_ULTRALYTICS=1` (offline CI only).
- tlc 3.0→3.1 compat shims (`evaluate.py` table_rows; `config.py` `register_url_alias`/`Configuration`).
- Reconciled prior godfire WIP: real 3.1 training migration (`.with_transform`, `create_sampler`),
  cleaner `evaluate.py`, `scripts/curation/balance_classes.py`.
- Self-verification harness: `benchmarks/baseline.json` + `scripts/eval/compare.py`.
- Full-trained both YOLO models on CUDA, re-baselined (old 0.975 → new 0.986/0.988).
- Fixed YOLO-seg train/val split to use the canonical seeded `chessvision-segmentation` split
  (`scripts/train/prepare_yolo_segmentation_dataset.py`).
- AGPL-3.0 relicense + LICENSE/NOTICE/README; checked in the two best models.
- CLAUDE.md + this roadmap.

## Active — start here next session

1. **Bigger + harder test set from the 600k uploads.** Two reasons, both important:
   - *Statistical power*: the current test set is **24 boards** — far too small to trust a
     1pp accuracy delta (today's old→new comparison is directionally good but not robust).
   - *Difficulty/coverage*: curate by the DATA_COLLECTION.md categories (lighting, angle,
     game state, environment), sampling diversely via 3LC embeddings (not by model failure).
   This unblocks every future "did X help?" question and is a strong 3LC selection demo.
2. **Merge PR #18** once reviewed (you're the author, so GitHub won't let you self-assign as
   reviewer — just review + merge, or add a 3LC colleague).
3. **Board-extraction failure analysis** on the new test set; treat the contour/quadrangle
   step as a suspect, not just the model.

## Next

- **Re-evaluate validation rules**: measure whether re-enabling king/bishop-count rules helps
  on the harder set; consider a constraint solver over greedy per-square fixes.
- **Coreset selection experiment** on the uploads: does a 3LC-selected coreset match full-data
  accuracy? Standalone 3LC demo.
- **Decouple metrics tests from tlc** (move pure functions out of `evaluate.py`) + add
  `integration` pytest markers so `-m "not integration"` is the fast offline lane.
- **Audit remaining 3.x-untested scripts**: `process_new_raw/*`, `merge_new_raw/*` (need S3).

## Later

- Model-assisted labeling loop for uploads (human-in-the-loop dataset growth).
- Active-learning loop driven by extraction-confidence / embedding outliers.
- Latency/packaging pass; revisit the Flask app + compute server.
- End-to-end FEN accuracy (not just per-square) as a headline metric.
- Board extraction via corner-keypoint regression (YOLO-pose): regress the 4 board
  corners directly instead of segment-mask → contour → perspective transform. Early spike
  in the closed #6 (`notebooks/convert_to_kpts.py` builds a keypoints table from existing
  masks via `_find_quadrangle`).

## Open questions for Gudbrand
- For the harder test set: label budget / how the 600k uploads' ground-truth FENs get created
  (model-assisted in the 3LC dashboard, then human-corrected?).

## Decisions log
- 2026-06-08: Claude takes code ownership; ROADMAP.md is the sync point. (Gudbrand)
- 2026-06-08: YOLO-only (UNet/timm fallback only); when goals diverge, Claude's call, flagged here. (Gudbrand)
- 2026-06-08: Use 3lc-ultralytics, not bare ultralytics; dev = editable `../tlc-monorepo` +
  `../3lc-ultralytics` + throwaway key with `TLC_DISABLE_ACCOUNT_SERVICE=1` (source honors it,
  the wheel doesn't). `pyproject` stays on public releases for external `uv sync`. (Gudbrand/verified)
- 2026-06-08: **License = AGPL-3.0 whole-repo**, and the two best models are committed.
  Rationale: built on Ultralytics (AGPL); weights inherit AGPL; goal is to stop proprietary
  commercial free-riding (getting paid isn't cleanly possible on Ultralytics). Reverses the
  earlier "don't commit weights / no fetch scripts" stance. (Gudbrand)
- 2026-06-08: Retrained both models on godfire, re-baselined on CUDA; fixed the YOLO-seg split
  to the canonical seeded split. Note: old→new (+1pp) is on a 24-board set — not statistically
  robust; a bigger test set is the prerequisite for confident model comparisons. (Claude)
