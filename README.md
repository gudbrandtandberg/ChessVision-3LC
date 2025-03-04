# ChessVision

A computer vision system for detecting and analyzing chess positions from images. The system uses deep learning models to segment chessboards and classify chess pieces.

## Features

- Chessboard detection and segmentation using a UNet model
- Chess piece classification using a deep learning model
- REST API for image processing and position analysis
- Web interface for uploading and analyzing chess images

## Project Structure

- `chessvision/`: Contains the core computer vision code, including model definitions and training scripts.
- `app/`: Contains the Flask web application code, including the REST API and web interface. Meant mainly for testing and development.
- `data/`: Contains the training and evaluation datasets.
- `weights/`: Contains the pre-trained models.

## Getting Started

+ Checkout this repo.
+ Ensure submodules are checked out: `git submodule update --init`
+ Create a virtual environment: `python -m venv .venv`
+ Activate the virtual environment: `source .venv/bin/activate`
+ Install dependencies: `pip install -r requirements.txt`

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

See launch configurations in `.vscode/launch.json` for training the models.
