# laneLineDetection/lane_detection_lib/image/resize.py
# Image resizing functions

from ..common import cv2, ImageType
from .io import get_aspect_ratio


def resize_by_width_height(img: ImageType, new_width: int,
                           new_height: int) -> ImageType:
    """Resize the image to the specified width and height."""

    if not isinstance(new_width, int) or not isinstance(new_height, int):
        raise ValueError("Width and height must be integers.")

    return cv2.resize(img, (new_width, new_height))


def resize_by_factor(img: ImageType, factor_x: float,
                     factor_y: float = None) -> ImageType:
    """Resize the image by a scaling factor."""

    if not isinstance(factor_x, (int, float)) or (
            factor_y is not None and not isinstance(factor_y, (int, float))):
        raise ValueError("Scaling factors must be numbers (int or float).")

    factor_y = factor_y if factor_y is not None else factor_x

    if factor_x <= 0 or factor_y <= 0:
        raise ValueError("Scaling factors must be positive numbers.")

    return cv2.resize(img, None, fx=factor_x, fy=factor_y)


def resize_by_aspect_ratio(img: ImageType, new_width: int = None,
                           new_height: int = None) -> ImageType:
    """Resize the image while maintaining its aspect ratio."""
    if new_width is not None and (
            not isinstance(new_width, int) or new_width <= 0):
        raise ValueError("new_width must be a positive integer.")

    if new_height is not None and (
            not isinstance(new_height, int) or new_height <= 0):
        raise ValueError("new_height must be a positive integer.")

    # Default values if none are given
    if new_width is None and new_height is None:
        new_width, new_height = img.shape[1] // 2, img.shape[0] // 2

    aspect_ratio = get_aspect_ratio(img)

    if new_width is not None:
        new_height = int(new_width / aspect_ratio)
    else:
        new_width = int(new_height * aspect_ratio)

    return cv2.resize(img, (new_width, new_height))
