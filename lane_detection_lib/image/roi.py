# lane_detection/lane_detection_lib/image/roi.py
# Region of Interest (ROI) functions
from typing import Optional

from ..common import Enum, cv2, np, ImageType
from .io import get_image_dimensions


class MaskType(Enum):
    """Enum for Mask Types."""
    triangle = 0
    square = 1
    circle = 2


class MaskColor(Enum):
    """Enum for Mask colors."""
    red = (255, 0, 0)
    blue = (0, 0, 255)
    green = (0, 255, 0)
    yellow = (255, 255, 0)
    white = (255, 255, 255)
    black = (0, 0, 0)


def find_center_image(height: int | float,
                      width: int | float) -> tuple[int, int]:
    """Find the center of an image using its shape."""
    if not isinstance(width, (int, float)) or not isinstance(
            height, (int, float)):
        raise TypeError('width and height must be integers')
    return height // 2, width // 2


def apply_roi_triangular(image: ImageType,
                         color: MaskColor = MaskColor.white) -> ImageType:
    """Apply a triangular ROI mask to the image."""
    height, width = get_image_dimensions(image)

    # Define the triangular region vertices
    vertices = np.array([[
        (int(width * 0.05), height),  # Bottom-left corner
        (int(width * 0.95), height),  # Bottom-right corner
        (int(width * 0.5), int(height * 0.6))  # Top-center
    ]], dtype=np.int32)

    # Ensure vertices array has the correct shape
    if (len(vertices.shape) != 3 or vertices.shape[1] < 3 or
            vertices.shape[2] != 2):
        raise ValueError(
            "Vertices array must have shape (1, n, 2) with n >= 3.")

    # Create a blank single-channel mask
    mask = np.zeros((height, width), dtype=np.uint8)

    # Fill the ROI on the mask
    cv2.fillPoly(mask, vertices, color.value)

    return mask


def apply_roi_rectangular(image: ImageType, star_point: tuple[int, int],
                          end_point: tuple[int, int], color: MaskColor,
                          thickness: int) -> ImageType:
    """Apply a rectangular ROI mask to the image."""
    height, width = get_image_dimensions(image)

    # Create a blank single-channel mask
    mask = np.zeros((height, width), dtype=np.uint8)

    # Draw a rectangle
    cv2.rectangle(mask, star_point, end_point, color.value, thickness)

    return mask


def apply_roi_circular(image: ImageType, center: tuple[int, int],
                       radius: int, color: MaskColor,
                       thickness: int) -> ImageType:
    """Apply a circular ROI mask to the image."""
    height, width = get_image_dimensions(image)

    # Create a blank single-channel mask
    mask = np.zeros((height, width), dtype=np.uint8)

    # Draw a circle
    cv2.circle(mask, center, radius, color.value, thickness)

    return mask


def apply_roi_mask(image: ImageType, mask_type: MaskType,
                   color: MaskColor = MaskColor.white,
                   thickness: int = 2,
                   center: Optional[tuple[int, int]] = None,
                   radius: Optional[int] = None,
                   start_point: Optional[tuple[int, int]] = None,
                   end_point: Optional[tuple[int, int]] = None) -> ImageType:
    """Applies a specified region of interest (ROI) mask to the image."""
    if mask_type == MaskType.triangle:
        return apply_roi_triangular(image, color)

    elif mask_type == MaskType.square:
        if start_point is None or end_point is None:
            raise ValueError('start_point and end_point must be specified')
        return apply_roi_rectangular(
            image, start_point, end_point, color, thickness)

    elif mask_type == MaskType.circle:
        if center is None or radius is None:
            raise ValueError('center and radius must be specified')
        return apply_roi_circular(image, center, radius, color, thickness)

    else:
        raise ValueError(f"Unsupported mask type: {mask_type}")
