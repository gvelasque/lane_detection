# lane_detection/lane_detection_lib/image/filter.py
# Image filtering functions

from ..common import cv2, np, ImageType


def apply_bilateral_filter(img: ImageType, diameter: int, sigma_color: float,
                           sigma_space: float) -> ImageType:
    """Apply bilateral filtering to the input image."""

    if not isinstance(diameter, int) or diameter <= 0 or diameter % 2 == 0:
        raise ValueError("Diameter must be an odd positive integer.")

    if not isinstance(sigma_color, (int, float)) or sigma_color <= 0:
        raise ValueError("Sigma color must be a positive number.")

    if not isinstance(sigma_space, (int, float)) or sigma_space <= 0:
        raise ValueError("Sigma space must be a positive number.")

    return cv2.bilateralFilter(img, diameter, sigma_color, sigma_space)


def filter_2d(img: ImageType, kernel: np.ndarray) -> ImageType:
    """Apply a 2D convolution filter to the image."""
    if not isinstance(kernel, np.ndarray):
        raise TypeError("Kernel must be a NumPy array.")
    if kernel.ndim != 2:
        raise ValueError("Kernel must be a 2D matrix.")

    return cv2.filter2D(img, -1, kernel)
