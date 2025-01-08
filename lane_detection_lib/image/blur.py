# lane_detection_lib/image/__init__.py
# Blurring functions

from ..common import cv2, ImageType


def validate_kernel_size(kernel_size) -> None:
    """Validate kernel size for filtering."""
    # Handle single integer case
    if isinstance(kernel_size, int):
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("Kernel size must be a positive odd integer.")
    # Handle tuple case
    elif isinstance(kernel_size, tuple):
        if not all(isinstance(x, int) and x > 0 and x % 2 == 1 for x in
                   kernel_size):
            raise ValueError(
                "Kernel size must be a tuple of positive odd integers.")
    else:
        raise TypeError(
            "Kernel size must be an integer or a tuple of integers.")
def apply_blur(img: ImageType, kernel_size: tuple[int, int]) -> ImageType:
    """Apply averaging blur to the input image."""
    validate_kernel_size(kernel_size)

    return cv2.blur(img, kernel_size)


def apply_gaussian_blur(img: ImageType, kernel_size: tuple[int, int],
                        deviation: int = 0) -> ImageType:
    """Apply Gaussian blur to the input image."""
    validate_kernel_size(kernel_size)

    return cv2.GaussianBlur(img, kernel_size, deviation)


def apply_median_blur(img: ImageType, kernel_size: int) -> ImageType:
    """Apply median blur to the input image."""
    validate_kernel_size(kernel_size)

    return cv2.medianBlur(img, kernel_size)



