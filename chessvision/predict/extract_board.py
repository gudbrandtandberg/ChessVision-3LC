"""
CNN-based board extraction.
Methods to extract boards from images.
"""

import cv2
import numpy as np
import torch

from ..utils import BOARD_SIZE, get_device, ratio

device = get_device()


def preprocess_image(
    image: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """
    Preprocess image for model input.

    Args:
        image: Input image as numpy array with shape (H, W, C)
        device: PyTorch device to place tensor on

    Returns:
        Preprocessed image batch as tensor with shape (1, C, H, W)
    """
    image_batch = torch.Tensor(np.array([image])) / 255
    image_batch = image_batch.permute(0, 3, 1, 2).to(device)
    return image_batch


def get_probabilities(
    model: torch.nn.Module,
    image_batch: torch.Tensor,
) -> np.ndarray:
    """
    Get probability mask from model, applying sigmoid if needed.

    Args:
        model: PyTorch model that outputs segmentation logits/probabilities
        image_batch: Batch of preprocessed images as tensor

    Returns:
        Probability mask as numpy array with shape (H, W)
    """
    with torch.no_grad():
        logits = model(image_batch)

    # Check if logits need sigmoid activation
    sample_values = logits[0].flatten()[:10].cpu().numpy()
    needs_sigmoid = np.any(sample_values < 0) or np.any(sample_values > 1)

    if needs_sigmoid:
        probabilities = torch.sigmoid(logits)
    else:
        probabilities = logits

    return probabilities[0].squeeze().cpu().numpy()


def process_board(
    orig: np.ndarray,
    approx: np.ndarray,
    board_size: tuple[int, int],
) -> np.ndarray:
    """
    Process extracted board to final format.

    Args:
        orig: Original input image
        approx: Quadrangle approximation of the board
        board_size: Target size for the output board (width, height)

    Returns:
        Processed board image as grayscale numpy array
    """
    board = extract_perspective(
        orig,
        approx,
        board_size,
    )
    if len(board.shape) == 3:  # If image has multiple channels
        board = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
    board = cv2.flip(board, 1)  # TODO: permute approximation instead..
    return board


def extract_board(
    image: np.ndarray,
    orig: np.ndarray,
    model: torch.nn.Module,
    threshold: float = 0.3,
) -> tuple[np.ndarray | None, np.ndarray]:
    """
    Extract chessboard from an image using a segmentation model.

    Args:
        image: Preprocessed input image (256x256)
        orig: Original input image at full resolution
        model: PyTorch model for board segmentation
        threshold: Probability threshold for mask binarization

    Returns:
        tuple containing:
            - Extracted board image or None if no board was found
            - Probability mask from the model
    """
    image_batch = preprocess_image(
        image,
        device,
    )
    probabilities = get_probabilities(
        model,
        image_batch,
    )
    mask = fix_mask(
        probabilities,
        threshold=threshold,
    )

    # approximate chessboard-mask with a quadrangle
    approx = find_quadrangle(mask)
    if approx is None:
        return None, probabilities

    # scale approximation to input image size
    approx = scale_approx(
        approx,
        (orig.shape[0], orig.shape[1]),
    )

    # extract board
    board = process_board(
        orig,
        approx,
        BOARD_SIZE,
    )

    return board, probabilities


def fix_mask(
    mask: np.ndarray,
    threshold: float = 0.3,
) -> np.ndarray:
    """
    Convert probability mask to binary mask.

    Args:
        mask: Probability mask as numpy array
        threshold: Threshold value for binarization

    Returns:
        Binary mask as uint8 numpy array
    """
    mask = mask.copy()  # Create a copy to avoid modifying the original
    mask[mask > threshold] = 255
    mask[mask <= threshold] = 0
    mask = mask.astype(np.uint8)
    return mask


def scale_approx(
    approx: np.ndarray,
    orig_size: tuple[int, int],
) -> np.ndarray:
    """
    Scale quadrangle approximation to match original image size.

    Args:
        approx: Quadrangle approximation from the 256x256 mask
        orig_size: Original image size (height, width)

    Returns:
        Scaled approximation as uint32 numpy array
    """
    sf = orig_size[0] / 256.0
    scaled = np.array(approx * sf, dtype=np.uint32)
    return scaled


def rotate_quadrangle(approx: np.ndarray) -> np.ndarray:
    """
    Rotate quadrangle to ensure consistent orientation.

    Args:
        approx: Quadrangle approximation

    Returns:
        Rotated quadrangle
    """
    if approx[0, 0, 0] < approx[2, 0, 0]:
        approx = approx[[3, 0, 1, 2], :, :]
    return approx


def find_quadrangle(mask: np.ndarray) -> np.ndarray | None:
    """
    Find a quadrangle (4-sided polygon) in a binary mask.

    Args:
        mask: Binary mask as uint8 numpy array

    Returns:
        Quadrangle approximation or None if no suitable quadrangle was found
    """
    try:
        contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
    except:
        _, contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)

    if len(contours) > 1:
        # print("Found {} contour(s)".format(len(contours)))
        contours = ignore_contours(mask.shape, contours)
        # print("Filtered to {} contour(s)".format(len(contours)))

    if len(contours) == 0:
        return None

    approx = None

    # try to approximate and hope for a quad
    for i in range(len(contours)):
        cnt = contours[i]

        arclen = cv2.arcLength(cnt, True)
        candidate = cv2.approxPolyDP(cnt, 0.1 * arclen, True)

        if len(candidate) != 4:
            continue

        approx = rotate_quadrangle(candidate)
        break

    return approx


def extract_perspective(
    image: np.ndarray,
    approx: np.ndarray,
    out_size: tuple[int, int],
) -> np.ndarray:
    """
    Extract a perspective-corrected region from an image.

    Args:
        image: Input image
        approx: Quadrangle approximation of the region to extract
        out_size: Target size for the output (width, height)

    Returns:
        Perspective-corrected image
    """
    w, h = out_size[0], out_size[1]

    dest = ((0, 0), (w, 0), (w, h), (0, h))

    approx = np.array(approx, np.float32)
    dest = np.array(dest, np.float32)

    coeffs = cv2.getPerspectiveTransform(approx, dest)

    return cv2.warpPerspective(image, coeffs, out_size)


def ignore_contours(
    img_shape: tuple[int, int],
    contours: list[np.ndarray],
    min_ratio_bounding: float = 0.6,
    min_area_percentage: float = 0.35,
    max_area_percentage: float = 1.0,
) -> list[np.ndarray]:
    """
    Filter contours based on area and aspect ratio criteria.

    Args:
        img_shape: Shape of the image (height, width)
        contours: list of contours to filter
        min_ratio_bounding: Minimum aspect ratio for bounding rectangle
        min_area_percentage: Minimum area as percentage of image area
        max_area_percentage: Maximum area as percentage of image area

    Returns:
        Filtered list of contours
    """
    ret = []
    mask_area = float(img_shape[0] * img_shape[1])

    for i in range(len(contours)):
        ca = cv2.contourArea(contours[i])
        ca /= mask_area
        if ca < min_area_percentage or ca > max_area_percentage:
            continue
        _, _, w, h = cv2.boundingRect(contours[i])
        if ratio(h, w) < min_ratio_bounding:
            continue
        ret.append(contours[i])

    return ret


if __name__ == "__main__":
    pass
