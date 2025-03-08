# ChessVision

Welcome to ChessVision: a computer vision system for detecting and classifying chess positions from images.
The system uses deep learning models to segment chessboards and classify chess pieces.

This project is an evolution of the original [ChessVision](https://github.com/ChessVision/ChessVision),
reimagined with the [3LC](https://3lc.ai) ecosystem in mind. It features a complete rewrite from Keras to PyTorch
and a thorough modernization of the codebase.

## Motivation

While detecting chess positions is certainly useful (and something I personally enjoy!),
this project serves a broader purpose as my experimental playground for exploring
machine learning systems end-to-end. It's where I learn, practice, and have fun
working with:

- Data collection and annotation
- Model training and evaluation
- Performance monitoring and analysis
- ML-powered web application development

ChessVision has become my ideal testbed for tackling these challenges. While there
might be more optimal solutions to specific problems, the real value lies in the
patterns and abstractions I discover along the way. Recently, I've also started
using it to develop and test features for the [3LC](https://3lc.ai) data platform,
which I build with my team at 3lc.ai.

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

- Checkout this repo.
- Ensure submodules are checked out: `git submodule update --init`
- Create a virtual environment: `python -m venv .venv`
- Activate the virtual environment: `source .venv/bin/activate`
- Install dependencies: `pip install -r requirements.txt`

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

See launch configurations in `.vscode/launch.json` for training the models and
running the web application.

## ChessVision Pipeline

- The chessvision solution: (see old repo README for details):
  - segmentation
  - thresholding
  - contour detection and filtering
  - chessboard extraction (from detected quad contour)
  - piece extraction
  - piece classification
  - position logic
  - final position output (FEN string)

- Original datasets: chessboard_segmentation, piece_classification, and test. Checked in to the repo.
- A practically endless supply of new data, collected through a friends chess app.

Pipelines for processing the new data: ...

## 3LC

![ChessVision Pipeline](examples/screenshots/embeddings.png)

![ChessVision Pipeline](examples/screenshots/new_raw_data.png)

![ChessVision Pipeline](examples/screenshots/piece_prediction.png)

![ChessVision Pipeline](examples/screenshots/run_overview.png)

![ChessVision Pipeline](examples/screenshots/test_results.png)


