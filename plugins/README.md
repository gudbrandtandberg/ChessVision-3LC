# ChessVision plugins for 3LC Insights

External [3lc-insights](https://github.com/3lc-ai) compute-service plugins, kept in this repo so
they can `import` ChessVision and live alongside the data they curate. Currently:

- **`chessvision_testset/`** — *ChessVision Test Set*: sample upload candidates → model-assisted
  draft FENs → human review → commit a test batch to 3LC tables (`chessvision-testing/test/<batch>`
  + the joined `test-all` tip). See `scripts/curation/{sample,predict}_test_candidates.py` and
  `scripts/curation/testset_tables.py` — the plugin orchestrates those.

## How it loads

The compute-service discovers *external* plugin dirs listed in its persistent settings file,
`~/.3lc-compute/settings.json`, under `plugin_dirs`. Each listed dir is added to `sys.path` and
its **subdirectories** are imported as packages, so point it at this `plugins/` dir:

```jsonc
// ~/.3lc-compute/settings.json
{
  "plugin_dirs": [
    "/home/gubbis/projects/ChessVision-3LC/plugins"
  ]
}
```

(Or, transiently, `TLC_COMPUTE_EXTERNAL_PLUGIN_DIRS=/home/gubbis/projects/ChessVision-3LC/plugins`,
or via the Hub Settings UI. Restart / reload the compute-service after editing the file.)

## Architecture: thin orchestrator

The plugin runs in the **compute-service venv**, but the heavy stages (sampling the LaCie pool,
running the YOLO pipeline, writing 3LC tables) are **shelled into this repo's `.venv`**
(`<repo>/.venv/bin/python -m scripts.curation.*`). That keeps the compute venv free of
torch/ultralytics and reuses ChessVision's exact environment. Only the lightweight review
endpoints (serve item, render FEN→SVG, record a decision) run in-process.

### Requirements
- This repo's `.venv` set up as usual (Tier 2 dev), reachable at `<repo>/.venv/bin/python`.
- In the **compute-service venv**: `pip install chess` (python-chess) — used to render FEN SVGs
  and validate edited FENs in-process. Nothing else.

## Notes / TODO
- Image refs in the committed tables use absolute paths; register a `CHESSVISION_TEST_ROOT`
  URL alias for portability if the hub ever runs on another host.
- Prediction currently runs via the bare-ultralytics offline path; switch to the 3lc-ultralytics
  path if you want lineage from the prediction run.
