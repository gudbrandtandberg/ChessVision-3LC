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
its **subdirectories** are scanned for a `plugin.toml` manifest; a host plugin is then loaded via
the manifest's `runtime.entrypoint` (`chessvision_testset:ChessVisionTestSetPlugin`). The manifest
is the single source of truth for the plugin's metadata — there is no `register()` call and no
metadata on the plugin class. Point `plugin_dirs` at this `plugins/` dir:

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

## Architecture: thin orchestrator (isolated venv)

This is a **`venv` plugin** (`isolation = "venv"` in `plugin.toml`): the compute-service never
imports it — a worker process runs it out-of-process in the plugin's **own uv venv**, and the
host reverse-proxies its routes / job channel. That venv is light: it holds only the compute-service
base (`3lc-compute` → worker harness + plugin_sdk + litestar + uvicorn) plus `python-chess`, so
`chess` stays **out of the main compute-service venv**.

The heavy stages (sampling the LaCie pool, running the YOLO pipeline, writing 3LC tables) are still
**shelled into this repo's `.venv`** (`<repo>/.venv/bin/python -m scripts.curation.*`), reusing
ChessVision's exact torch/ultralytics/tlc environment. Only the lightweight review endpoints (serve
item, render FEN→SVG, record a decision) run in-process in the worker.

The plugin's entrypoint is a **flat module** (`chessvision_testset_plugin.py`, not a package): the
worker launches with the plugin dir as `cwd` and adds it to `sys.path`, so the module resolves
directly while staying in the repo tree (so its `<repo>`-relative paths keep working).

### Requirements & provisioning
- This repo's `.venv` set up as usual (Tier 2 dev), reachable at `<repo>/.venv/bin/python` — used
  for the heavy stages.
- The **plugin venv** is built from `chessvision_testset/pyproject.toml`. Provision it with
  `uv sync` in `chessvision_testset/` (creates `chessvision_testset/.venv`), or let the Hub
  lazy-provision it on first use (Plugins page → provision). No `pip install` into the main venv.
- Dev note: `pyproject.toml` resolves `3lc-compute` editable from the local compute-service checkout
  (`../../../3lc-insights/compute-service`); in production it installs from the published index.

## Notes / TODO
- Image refs in the committed tables use absolute paths; register a `CHESSVISION_TEST_ROOT`
  URL alias for portability if the hub ever runs on another host.
- Prediction currently runs via the bare-ultralytics offline path; switch to the 3lc-ultralytics
  path if you want lineage from the prediction run.
