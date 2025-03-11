# ChessVision

Welcome to ChessVision: a computer vision system for detecting and classifying chess positions from images of 2D chessboards.
The system uses deep learning models to segment chessboards and classify chess pieces.

This project is an evolution of the original [ChessVision](https://github.com/ChessVision/ChessVision),
reimagined with the [3LC](https://3lc.ai) integrated in all stages of the pipeline. It also features a complete rewrite in PyTorch and a thorough upgrade of the codebase.

## Motivation

While detecting chess positions is certainly useful (and something I personally enjoy!),
this project serves a broader purpose as my experimental playground for exploring
machine learning systems. It's where I learn, practice, and have fun
working with:

- Data collection and annotation
- Model training and evaluation
- Performance monitoring and analysis

The [3LC](https://3lc.ai) data platform, built by the incredible team at 3LC AI, serves as the perfect tool for data collection, data selection, labeling, and monitoring the performance of the system.

## Features

- Chessboard segmentation using a UNet model
- Chess piece classification using either a YOLO model, or a timm-based model
- API for image processing
- Web interface for uploading and analyzing chess images

## Project Structure

- `chessvision/`: The core computer vision code
- `scripts/`: Scripts for training and evaluating the models
- `app/`: Code for a development Flask web application and compute server
- `data/`: Training and evaluation datasets
- `weights/`: Pre-trained models (not included in the repo - train your own models or contact me for a copy)
- `tests/`: Unit tests for the computer vision code

## Installation

### 1. Clone the Repository

```bash
# Clone the repo
git clone https://github.com/gudbrandtandberg/ChessVision-3LC.git
cd ChessVision-3LC

# Initialize and update the pytorch-unet submodule
git submodule update --init
```

### 2. Create Virtual Environment

```bash
# Create a new virtual environment
python -m venv .venv

# Activate the environment
# On Windows (Git Bash):
source .venv/Scripts/activate
# On Unix/MacOS:
source .venv/bin/activate
```

### 3. Install Dependencies

Choose either approach:

#### Option A: Using uv (Recommended - Faster)
```bash
# Install uv
pip install uv

# Install dependencies
uv pip install -e ".[dev,viz,yolo]"
```

#### Option B: Using pip
```bash
# Install in editable mode with all dependency groups
pip install -e ".[dev,viz,yolo]"
```

### 4. Set up 3LC

1. Get your API key:
   - Go to [accounts.3lc.ai](https://accounts.3lc.ai)
   - Copy your API key

2. Login to 3LC:
```bash
# Login with your API key
3lc login

# Verify installation
3lc --version
```

## Verify Installation

```bash
# Run a simple test to verify everything is working
python -c "from chessvision import ChessVision; print('Installation successful!')"

# Verify PyTorch installation
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

## Examples

### Quick Start

For a quick end-to-end overview of the system, run the Jupyter notebook:

```bash
jupyter notebook examples/quickstart-example.ipynb
```

### Detailed Pipeline

For a detailed breakdown of each step in the pipeline, run the Python script:

```bash
python examples/detailed-example.py
```

## Training

Main scripts for training and evaluating the models are located in the `scripts/` directory.

Both models usually take less than 10 minutes to train on a modern GPU.

See additional launch configurations in `.vscode/launch.json` for training the models and
running the web application.

## The ChessVision Solution

The chessvision solution consists of several steps:

1. Board detection using a UNet model to segment the chessboard from the background
2. Contour detection and filtering to identify the chessboard boundaries
3. Perspective transform to extract the chessboard
4. Individual square extraction
5. CNN-based piece classification
6. Chess position validation and final output (FEN string)

For more details on the original approach, see the [old README](https://github.com/gudbrandtandberg/ChessVision?tab=readme-ov-file#algorithm-details)

## Datasets

The repo comes with three original datasets checked in:

- `board_extraction`: A dataset of chessboard images with annotated segmentation masks.
- `squares`: A dataset of chess piece images with annotated classification labels.
- `test`: A set of test images with ground truth files for evaluating the model.

In addition, there is a practically endless supply of new data collected through a friend's chess app, which I have in a private S3 bucket.

## 3LC and the ML Lifecycle

### Training the board extractor

![ChessVision Pipeline](examples/screenshots/run_overview.png)

### Training the piece classifier

![ChessVision Pipeline](examples/screenshots/piece_prediction.png)

### Process new raw data

![ChessVision Pipeline](examples/screenshots/embeddings.png)

![ChessVision Pipeline](examples/screenshots/new_raw_data.png)

### Run evaluation suite

![ChessVision Pipeline](examples/screenshots/test_results.png)
