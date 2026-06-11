# ChessVision Roadmap & Planning

> Living sync doc between Gudbrand and Claude. Claude owns the code; Gudbrand sets
> direction. Keep this short and current — prune done items, promote ideas as they ripen.

Last updated: 2026-06-10

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
- **Branch/PR**: foundation PR #18 + docs PR #19 are **merged** to `main`. GPU node
  `gubbis@godfire` is set up with a standalone `.venv`.
- **Provenance ledger** (`data/provenance/ledger.json`): single source of truth for which
  upload UUID belongs to which split (`train_val`/`test`/`candidate`/`rejected`). Sampling
  always excludes everything in it, so train/test can't leak. Seeded from on-disk splits
  (631 extraction `train_val` + 35 existing `test`).

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

1. **Bigger + harder test set from the 695k uploads** — *in progress*. Tooling built
   (2026-06-10), human review pending. Pipeline (all pure-Python, no `tlc`, in
   `scripts/curation/`):
   - `sample_test_candidates.py` → stratified-by-month seeded sample of 260 candidates
     into `data/test/harder-v1/` (oversampled to absorb rejects; targets ~200 keepers).
   - `predict_test_candidates.py` → model-drafts both orientations' FENs (259/260 boards
     extracted; 1 extraction failure).
   - `review_tool.py` → local Flask app, raw photo vs. rendered predicted FEN, keystroke
     accept/flip/edit/reject; writes confirmed FENs to `ground_truth/`, stamps the ledger.
   - **NEXT human step (Gudbrand)**: run the review tool and confirm/correct ~200 FENs.
   - *Deferred refinement*: visual-category diversity via embeddings (the 3LC selection
     demo). Current sampling is temporal-only — good spread across collection periods, but
     doesn't explicitly cover lighting/angle/game-state categories. Revisit after v1 lands.
2. **Re-baseline on `harder-v1`** once labeled: rerun `scripts/eval/compare.py`, expect
   accuracy to drop from the (inflated, 24-board) 0.986 — that's the point.
3. **Board-extraction failure analysis** on the new test set; treat the contour/quadrangle
   step as a suspect, not just the model. (1 hard failure already in `harder-v1`; inspect it.)

## Next

- **Re-evaluate validation rules**: measure whether re-enabling king/bishop-count rules helps
  on the harder set; consider a constraint solver over greedy per-square fixes.
- **Coreset selection experiment** on the uploads: does a 3LC-selected coreset match full-data
  accuracy? Standalone 3LC demo.
- **Decouple metrics tests from tlc** (move pure functions out of `evaluate.py`) + add
  `integration` pytest markers so `-m "not integration"` is the fast offline lane.
- **Audit remaining 3.x-untested scripts**: `process_new_raw/*`, `merge_new_raw/*` (need S3).
- **Periodically verify the public-wheel path**: run the suite against a clean `uv sync` (no
  source overlay) + a real free key. That's the default external-user experience (Tier 1 in
  CONTRIBUTING) and is easy to let rot while we live on the source overlay.

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
- *(resolved 2026-06-10)* Harder-set labeling: model-assisted drafts + human correction via a
  throwaway local tool (bypassing 3LC), ~200 boards. See decisions log.
- Should `harder-v1`'s raw photos (~22 MB, 260 imgs) be committed to the repo like
  `test/initial`, or kept out (LFS / external)? Defaulted to in-repo for now (matches
  precedent, sizes comparable). Flag if you'd rather not grow the repo.

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
- 2026-06-11: **Test data as 3LC tables + Insights plugin.** Test batches are materialized as
  3LC tables (`scripts/curation/testset_tables.py`): one table per batch under
  `chessvision-testing/test/<slug>` (cols `image`,`fen`,`source`,`month`), joined into a rebuilt
  `test-all` tip; files+ledger stay the source of truth, tables are derived/reproducible. The
  add-batch workflow is wrapped as an **external 3lc-insights compute-service plugin** living in
  this repo at `plugins/chessvision_testset/` (loaded via `~/.3lc-compute/settings.json`
  `plugin_dirs`). Plugin is a thin orchestrator: shells heavy stages into this repo's `.venv`,
  serves review in-process. Decision: keep the plugin in this repo, not in 3lc-insights. Eval now
  reads ground truth from the table's `fen` column (sidecar `.txt` fallback for legacy tables),
  and `evaluate_model` returns the metrics dict with 3LC **Run creation opt-in** (`--create-run` /
  `--include-metrics-table`); `compare.py` runs dashboard-free. (Gudbrand/Claude)
- 2026-06-10: **Harder test set sourcing.** Raw uploads live on the LaCie disk at
  `/media/gubbis/LACIE/chessvision-raw-uploads` (~695k UUID-named JPGs, foldered `YYYY/MM/DD`).
  Sourcing discipline enforced by a persistent **provenance ledger** keyed on UUID — sampling
  excludes anything already assigned to a split, so no train/test leakage. Target ~200 boards.
  Labeling = **model-assisted draft + human correction via a throwaway local Flask tool,
  bypassing 3LC** (render predicted FEN as SVG beside the raw photo, keystroke confirm). Chose
  this over the 3LC-dashboard route for speed/simplicity; the embedding-based *selection* demo
  is deferred, not abandoned. (Gudbrand/Claude)
