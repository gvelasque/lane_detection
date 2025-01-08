# laneLineDetection/lane_detection_lib/image/crop.py
# Cropping functions

from ..common import ImageType
from .io import get_image_dimensions

def crop_image(image: ImageType, y_start: int, y_end: int, x_start: int,
               x_end: int) -> ImageType:
    """Crop the image using the specified coordinates."""

    if not all(isinstance(i, int) for i in [y_start, y_end, x_start, x_end]):
        raise ValueError("Coordinates must be integers.")

    height, width = get_image_dimensions(image)

    if not (0 <= y_start < y_end <= height and 0 <= x_start < x_end <= width):
        raise ValueError(
            "Invalid crop dimensions. Ensure 0 ≤ "
            "start < end and within image bounds.")

    return image[y_start:y_end, x_start:x_end]
