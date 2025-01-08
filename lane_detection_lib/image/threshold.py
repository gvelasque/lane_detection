# laneLineDetection/lane_detection_lib/image/threshold.py
# Thresholding techniques

from ..common import cv2, np, ImageType
from .color_conversion import convert_to_rgb2grayscale


def apply_threshold(
        img: ImageType, thresh: int = 127, maxval: int = 255) -> ImageType:
    """Apply binary thresholding to the input image."""

    if not isinstance(thresh, int) or not (0 <= thresh <= 255):
        raise ValueError("Threshold must be an integer between 0 and 255.")

    if not isinstance(maxval, int) or not (0 <= maxval <= 255):
        raise ValueError("Max value must be an integer between 0 and 255.")

    # Ensure grayscale and convert
    if img.ndim == 3:
        if img.shape[2] == 3:
            img = convert_to_rgb2grayscale(img)
        elif img.shape[2] == 1:
            #  If single channel stored in 3D, reshape
            img = img.reshape(img.shape[0], img.shape[1])

    # Apply binary thresholding
    _, binary_image = cv2.threshold(img, thresh, maxval, cv2.THRESH_BINARY)

    return binary_image


def apply_adaptive_threshold(img: ImageType, max_value: int = 255,
                             block_size: int = 11, c: int = 2) -> np.ndarray:
    """Apply adaptive thresholding to the image."""

    if not isinstance(max_value, int) or not (0 <= max_value <= 255):
        raise ValueError("max_value must be an integer between 0 and 255.")

    if not isinstance(
            block_size, int) or block_size % 2 == 0 or block_size <= 1:
        raise ValueError(
            "block_size must be an odd positive integer greater than 1.")

    if not isinstance(c, int):
        raise ValueError("C must be an integer.")

    # Ensure grayscale and convert
    if img.ndim != 2:
        img = convert_to_rgb2grayscale(img)

    return cv2.adaptiveThreshold(
        img, max_value, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, block_size, c
    )
