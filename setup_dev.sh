#!/usr/bin/env bash
source .venv/Scripts/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 --force-reinstall
uv pip install -e ../tlc-monorepo[pacmap]