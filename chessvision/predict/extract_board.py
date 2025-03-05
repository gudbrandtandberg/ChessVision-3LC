"""
CNN-based board extraction.
Methods to extract chess boards from images.
"""

from dataclasses import dataclass

import cv2
import numpy as np
import torch

from ..utils import BOARD_SIZE, get_device, ratio


@dataclass
class ExtractionResult:
    """Results from board extraction process."""

    board_image: np.ndarray | None  # The extracted board image, or None if no board found
    probability_mask: np.ndarray  # The probability mask from the model
    quadrangle: np.ndarray | None  # The detected quadrangle, or None if no board found
    confidence: float  # Confidence score for the extraction


class BoardExtractor:
    """Chess board extractor using deep learning segmentation."""

    def __init__(self, model: torch.nn.Module | None = None):
        """
        Initialize the board extractor.

        Args:
            model: PyTorch model for board segmentation. If None, a model must be provided
                  when calling extract_board.
        """
        self.device = get_device()
        self.model = model

    def extract_board(
        self,
        image: np.ndarray,
        orig_image: np.ndarray,
        model: torch.nn.Module | None = None,
        threshold: float = 0.3,
    ) -> ExtractionResult:
        """
        Extract chessboard from an image using a segmentation model.

        Args:
            image: Preprocessed input image (256x256)
            orig_image: Original input image at full resolution
            model: PyTorch model for board segmentation (overrides the one set in constructor)
            threshold: Probability threshold for mask binarization

        Returns:
            ExtractionResult containing the extracted board and metadata

        Raises:
            ValueError: If no model is provided either in constructor or as parameter
        """
        model_to_use = model or self.model
        if model_to_use is None:
            raise ValueError("No model provided for board extraction")

        # Process image and get probability mask
        image_batch = self._preprocess_image(image)
        probabilities = self._get_probabilities(model_to_use, image_batch)

        # Create binary mask and find board
        mask = self._fix_mask(probabilities, threshold)
        quadrangle = self._find_quadrangle(mask)

        # Calculate confidence score
        confidence = self._calculate_confidence(probabilities, mask)

        # If no board found, return early
        if quadrangle is None:
            return ExtractionResult(
                board_image=None,
                probability_mask=probabilities,
                quadrangle=None,
                confidence=confidence,
            )

        # Scale approximation to input image size
        scaled_quad = self._scale_approx(quadrangle, (orig_image.shape[0], orig_image.shape[1]))

        # Extract and process board
        board = self._process_board(orig_image, scaled_quad, BOARD_SIZE)

        return ExtractionResult(
            board_image=board,
            probability_mask=probabilities,
            quadrangle=scaled_quad,
            confidence=confidence,
        )

    def _calculate_confidence(self, probabilities: np.ndarray, mask: np.ndarray) -> float:
        """Calculate confidence score for the extraction."""
        # Use the top 25% most confident predictions
        flat_probs = probabilities.flatten()
        k = int(flat_probs.size * 0.25)
        sorted_probs = np.sort(flat_probs)[-k:]
        confidence = np.mean(np.abs(sorted_probs - 0.5)) * 2
        return float(confidence)

    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model input."""
        image_batch = torch.Tensor(np.array([image])) / 255
        image_batch = image_batch.permute(0, 3, 1, 2).to(self.device)
        return image_batch

    def _get_probabilities(self, model: torch.nn.Module, image_batch: torch.Tensor) -> np.ndarray:
        """Get probability mask from model, applying sigmoid if needed."""
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

    def _fix_mask(self, mask: np.ndarray, threshold: float = 0.3) -> np.ndarray:
        """Convert probability mask to binary mask."""
        mask = mask.copy()  # Create a copy to avoid modifying the original
        mask[mask > threshold] = 255
        mask[mask <= threshold] = 0
        mask = mask.astype(np.uint8)
        return mask

    def _scale_approx(self, approx: np.ndarray, orig_size: tuple[int, int]) -> np.ndarray:
        """Scale quadrangle approximation to match original image size."""
        sf = orig_size[0] / 256.0
        scaled = np.array(approx * sf, dtype=np.uint32)
        return scaled

    def _rotate_quadrangle(self, approx: np.ndarray) -> np.ndarray:
        """Rotate quadrangle to ensure consistent orientation."""
        if approx[0, 0, 0] < approx[2, 0, 0]:
            approx = approx[[3, 0, 1, 2], :, :]
        return approx

    def _find_quadrangle(self, mask: np.ndarray) -> np.ndarray | None:
        """Find a quadrangle (4-sided polygon) in a binary mask."""
        try:
            contours, _ = cv2.findContours(
                mask,
                cv2.RETR_CCOMP,
                cv2.CHAIN_APPROX_TC89_KCOS,
            )
        except:
            _, contours, _ = cv2.findContours(
                mask,
                cv2.RETR_CCOMP,
                cv2.CHAIN_APPROX_TC89_KCOS,
            )

        if len(contours) > 1:
            contours = self._ignore_contours(mask.shape, contours)

        if len(contours) == 0:
            return None

        approx = None

        # try to approximate and hope for a quad
        for i in range(len(contours)):
            cnt = contours[i]

            arclen = cv2.arcLength(cnt, True)
            candidate = cv2.approxPolyDP(
                cnt,
                0.1 * arclen,
                True,
            )

            if len(candidate) != 4:
                continue

            approx = self._rotate_quadrangle(candidate)
            break

        return approx

    def _process_board(
        self,
        orig: np.ndarray,
        approx: np.ndarray,
        board_size: tuple[int, int],
    ) -> np.ndarray:
        """Process extracted board to final format."""
        board = self._extract_perspective(
            orig,
            approx,
            board_size,
        )
        if len(board.shape) == 3:  # If image has multiple channels
            board = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
        board = cv2.flip(board, 1)  # TODO: permute approximation instead..
        return board

    def _extract_perspective(
        self,
        image: np.ndarray,
        approx: np.ndarray,
        out_size: tuple[int, int],
    ) -> np.ndarray:
        """Extract a perspective-corrected region from an image."""
        w, h = out_size[0], out_size[1]

        dest = ((0, 0), (w, 0), (w, h), (0, h))

        approx = np.array(approx, np.float32)
        dest = np.array(dest, np.float32)

        coeffs = cv2.getPerspectiveTransform(approx, dest)

        return cv2.warpPerspective(
            image,
            coeffs,
            out_size,
        )

    def _ignore_contours(
        self,
        img_shape: tuple[int, int],
        contours: list[np.ndarray],
        min_ratio_bounding: float = 0.6,
        min_area_percentage: float = 0.35,
        max_area_percentage: float = 1.0,
    ) -> list[np.ndarray]:
        """Filter contours based on area and aspect ratio criteria."""
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


# For backward compatibility
def extract_board(
    image: np.ndarray,
    orig: np.ndarray,
    model: torch.nn.Module,
    threshold: float = 0.3,
) -> tuple[np.ndarray | None, np.ndarray]:
    """
    Legacy function for backward compatibility.

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
    extractor = BoardExtractor(model)
    result = extractor.extract_board(image, orig, threshold=threshold)
    return result.board_image, result.probability_mask


if __name__ == "__main__":
    pass
