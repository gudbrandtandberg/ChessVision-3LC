# ChessVision Roadmap & Planning

> Living sync doc between Gudbrand and Claude. Claude owns the code; Gudbrand sets
> direction. Keep this short and current — prune done items, promote ideas as they ripen.

Last updated: 2026-06-08 (initial draft by Claude)

## The two goals

1. **Do well at the end-to-end task**: image → FEN. Board segmentation → extraction →
   square classification → position validation. Push accuracy, robustness (harder test
   set, better extraction), and latency.
2. **Be a great 3LC demo/dev playground**: every stage should showcase a real 3LC
   workflow (data selection, model-assisted labeling, metrics analysis, lineage, sample
   weighting, coreset selection, embeddings). The ~600k user uploads are raw fuel for this.

These pull in mostly the same direction: better data work → better model → better demo.

## Current state (as of handover)

- **Pipeline**: UNet *or* YOLO-seg for board extraction; YOLO-cls *or* timm/ResNet for
  pieces; OpenCV contour→perspective transform between them; chess-rule validation on top
  (only the "no pawns on rank 1/8" rule is active — king/bishop-count rules are commented out).
- **3LC integration**: training table creation, eval runs with rich metrics tables,
  new-raw-data enrichment pipeline (S3 download → table → board-extraction metrics +
  embeddings → pacmap), test-set merge pipeline.
- **Datasets in repo**: `board_extraction` (23M), `squares` (43M), `test` (3.7M).
- **Tests**: 7 high-level + metrics tests. Coverage of `core.py` is low (~20%) because most
  tests need trained weights.

## Decisions (2026-06-08, from Gudbrand)
- **YOLO-only**: stick to YOLO seg + cls models; they work well. UNet stays only as a
  fallback. No UNet weights maintained.
- **Weights are training output**, not artifacts to sync: "current best models" are produced
  by running the training scripts, not committed (Ultralytics AGPL caution). No fetch scripts.
- **cairo is an allowed dependency** — one-time install. Already present via brew; just gate
  imports so it can't break unrelated code.
- **3LC key + both editables**: intended local setup is `uv sync` then
  `pip install -e ../tlc-monorepo -e ../3lc-ultralytics` with a valid key. We use
  `3lc-ultralytics`, **not** bare ultralytics — bare loses the 3LC integration. A valid key is
  acceptable to require for the 3LC-backed paths (eval, training, new-raw). Bare ultralytics is
  opt-in only via `CHESSVISION_ALLOW_BARE_ULTRALYTICS=1` for offline CI/tests.

## Known friction / "claudability" gaps

- **C-1 — cairo (mostly resolved).** cairo is installed; Python just needs
  `DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib`. Documented in CLAUDE.md. Remaining: a
  couple of `tlc`-coupled modules still import-fail at collection (see C-2).
- **C-2 — `tlc` API key (resolved for local dev).** The frozen wheel hard-validates the key at
  import; the **source** install honors `TLC_DISABLE_ACCOUNT_SERVICE=1`, so editable
  `../tlc-monorepo` + `../3lc-ultralytics` + a throwaway key runs the **full** suite + eval
  locally with no real key. Bare ultralytics remains an opt-in escape hatch
  (`CHESSVISION_ALLOW_BARE_ULTRALYTICS=1`) for the no-source/no-key case. Optional polish:
  move the pure metric functions out of `scripts/eval/evaluate.py` so they don't need tlc at all.
- **C-3 — Runnable state (resolved by decision).** YOLO weights present; defaults now YOLO.
  Reaching a fresh runnable state = run the training scripts. No sync scripts.
- **C-4 — No baseline metrics in the repo.** Eval results live only in the 3LC dashboard.
  → Commit `benchmarks/baseline.json` (accuracy, validated-accuracy, extraction-failures,
  latency) + a compare script. Highest-value remaining item for self-verification. *(needs a
  valid key to generate the first baseline.)*
- **C-5 — Project map (done).** `CLAUDE.md` added.

## Now / Next / Later

### Now (foundation — make it claudable)
- [x] **Default to YOLO** for board extraction (was UNet → crashed on missing weights);
      fixed two real bugs exposed by it: non-contiguous tensor `.view()` crash and half→float32
      dtype mismatch in `extract_board`. Core tests green offline.
- [x] **YOLO loaders prefer 3lc-ultralytics, fail loud** — raise a clear error if it can't
      import (usually a bad key) instead of silently using bare ultralytics; bare is opt-in via
      `CHESSVISION_ALLOW_BARE_ULTRALYTICS=1` for offline CI/tests only. (C-2 partial)
- [x] **CLAUDE.md** with golden-path commands and gotchas. (C-5)
- [x] **Baseline + compare**: `benchmarks/baseline.json` (committed) + `scripts/eval/compare.py`
      which runs eval and diffs vs baseline, gating on accuracy regressions. (C-4)
      Initial set: top-1 0.982 / validated 0.983 / 0 extraction failures.
- [~] **Audit scripts for tlc 3.0→3.1 API skew** (we run monorepo 3.1; public release is 3.0,
      so fixes are version-compat shims, not hard switches):
      - [x] `evaluate.py`: `._tlc_url` → `table.table_rows`.
      - [x] `train/config.py`: `tlc.register_url_alias` → `tlc.url.*`; `tlc.Configuration` →
            `tlc.config` (both shimmed). This was blocking *all* training imports.
      - [x] `create_classification_tables.py` / `create_board_extraction_tables.py`: build
            clean on 3.1 (verified locally — 8931/2134 and 568/63 rows).
      - [x] YOLO train scripts import-clean (`tlc.active_run/schemas/helpers` all present in 3.1).
      - [ ] Remaining (need GPU/S3 to exercise): training run loops, `process_new_raw/*`,
            `merge_new_raw/*`, and the UNet/timm trainers (lower priority — YOLO-only).
- [ ] **Decouple metrics tests from tlc**: move pure functions (`board_to_labels`,
      `compute_model_topk_accuracy`, …) out of `scripts/eval/evaluate.py` into a tlc-free
      module so `test_metrics.py` collects offline. (C-2)
- [ ] **Pytest markers**: `integration` for model/3LC/render tests so `-m "not integration"`
      is the fast offline lane.

### Next (model & data quality)
- [ ] **Harder test set** from the 600k uploads, curated by the DATA_COLLECTION.md
      categories (lighting, angle, game state, environment). Use 3LC embeddings to sample
      diversely, not by model failure. Demoable selection workflow.
- [ ] **Better board extraction**: failure analysis on the new test set; compare UNet vs
      YOLO-seg head-to-head; investigate the contour/quadrangle step as a failure source.
- [ ] **Re-evaluate validation rules**: measure whether re-enabling king/bishop-count rules
      (currently commented out) helps or hurts on the harder set; consider a proper
      constraint solver instead of greedy per-square fixes.
- [ ] **Coreset selection experiment** on the uploads: does a 3LC-selected coreset match
      full-data accuracy? Strong standalone 3LC demo.

### Later (reach)
- [ ] Model-assisted labeling loop for uploads → grow training set with human-in-the-loop.
- [ ] Active-learning loop driven by extraction-confidence / embedding outliers.
- [ ] Latency/packaging pass; revisit the Flask app + compute server.
- [ ] End-to-end FEN accuracy (not just per-square) as a headline metric.

## Open questions for Gudbrand
- (none blocking — local setup works with a throwaway key. Will surface new ones as they arise.)

## Decisions log
- 2026-06-08: Claude takes code ownership; this doc is the sync point. (Gudbrand)
- 2026-06-08: YOLO-only; weights are training output (not committed); cairo allowed; valid
  3LC key acceptable for 3LC-backed paths. (Gudbrand) — see Decisions section above.
- 2026-06-08: Flipped board-extractor default to YOLO; fixed contiguous + dtype bugs; made
  YOLO loaders offline-resilient; added CLAUDE.md. Core suite green offline. (Claude)
- 2026-06-08: When the two goals diverge, Claude trades off case-by-case and flags it here.
  (Gudbrand)
- 2026-06-08: Use 3lc-ultralytics, not bare ultralytics; dev installs both `../tlc-monorepo`
  and `../3lc-ultralytics` editable. Loaders fail loud on missing 3LC; bare is opt-in for
  CI/tests only. (Gudbrand)
- 2026-06-08: Local dev runs `tlc` from source with `TLC_DISABLE_ACCOUNT_SERVICE=1` + throwaway
  key (no real key needed). `pyproject.toml` stays on public releases so external users can
  `uv sync`; editable installs are a local overlay, not committed. Full suite green this way. (verified)
- 2026-06-08: Committed first metrics baseline + `compare.py`; fixed a tlc 3.1 API break in
  `evaluate.py` (`._tlc_url` → `table_rows`). (Claude)
