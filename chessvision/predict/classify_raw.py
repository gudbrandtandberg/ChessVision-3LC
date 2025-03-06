import logging

import cv2
import torch

from chessvision.piece_classification.train_classifier import get_classifier_model
from chessvision.piece_classification.train_classifier import load_checkpoint as load_classifier_checkpoint
from chessvision.predict.classify_board import classify_board
from chessvision.predict.extract_board import BoardExtractor, extract_board
from chessvision.pytorch_unet.unet.unet_model import UNet
from chessvision.utils import (
    INPUT_SIZE,
    best_classifier_weights,
    best_extractor_weights,
    get_device,
)

logger = logging.getLogger("chessvision")

board_model: torch.nn.Module | None = None
sq_model: torch.nn.Module | None = None


def load_extractor_checkpoint(model: torch.nn.Module, checkpoint_path: str):
    """Load a model checkpoint with backward compatibility support."""
    device = get_device()
    state_dict = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        # New format
        model.load_state_dict(state_dict["model_state_dict"])
        metadata = state_dict.get("metadata", {})
    else:
        # Old format - state_dict is directly the model weights
        # or has "model" key instead of "model_state_dict"
        if "model" in state_dict:
            model_weights = state_dict["model"]
            metadata = state_dict.get("metadata", {})
        else:
            model_weights = state_dict
            metadata = {}
        model.load_state_dict(model_weights)

    if metadata:
        model.metadata = metadata

    return model


def load_classifier(checkpoint_path: str = best_classifier_weights):
    global sq_model

    if sq_model is not None:
        return sq_model

    device = get_device()
    classifier = get_classifier_model()
    classifier, _, _, _ = load_classifier_checkpoint(classifier, None, checkpoint_path)
    classifier.eval()
    classifier.to(device)

    return classifier


def load_board_extractor(checkpoint_path: str = best_extractor_weights):
    """Load the board extraction model."""
    global board_model

    if board_model is not None:
        return board_model

    device = get_device()
    extractor = UNet(n_channels=3, n_classes=1)
    extractor = extractor.to(memory_format=torch.channels_last)
    extractor = load_extractor_checkpoint(extractor, checkpoint_path)
    extractor.eval()
    extractor.to(device)
    board_model = extractor

    return extractor


def classify_raw(
    img,
    filename="",
    board_model=None,
    sq_model=None,
    flip=False,
    threshold=0.3,
):
    logger.debug(f"Processing image {filename}")

    if not board_model:
        board_model = load_board_extractor()
    if not sq_model:
        sq_model = load_classifier()

    ###############################   STEP 1    #########################################

    ## Resize image
    comp_image = cv2.resize(img, INPUT_SIZE, interpolation=cv2.INTER_AREA)

    ## Extract board using CNN model and contour approximation
    logger.debug("Extracting board from image")
    try:
        extractor = BoardExtractor(board_model)
        result = extractor.extract_board(comp_image, img, threshold=threshold)
        board_img = result.board_image
        # Return binary_mask for visualization, keep probabilities for metrics
        mask = result.binary_mask
    except Exception as e:
        raise e

    if board_img is None:
        return None, mask, None, None, None, None, None
    ###############################   STEP 2    #########################################

    logger.debug("Classifying squares")
    FEN, predictions, chessboard, squares, names = classify_board(board_img, sq_model, flip=flip)

    logger.debug(f"Processing image {filename}.. DONE")

    return board_img, mask, predictions, chessboard, FEN, squares, names
