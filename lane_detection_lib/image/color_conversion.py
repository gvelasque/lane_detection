# laneLineDetection/lane_detection_lib/image/color_conversion.py
# Color space conversions

from ..common import cv2, ImageType


def convert_to_rgb2grayscale(img: ImageType) -> ImageType:
    """Convert a RGB image to grayscale."""
    # if img.ndim != 3 or img.shape[2] != 3:
    #    raise ValueError("Input image must be a 3-channel (BGR) color image.")
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def convert_bgr2rgb(img: ImageType) -> ImageType:
    """Convert a BGR image to RGB format."""
    # if img.ndim != 3 or img.shape[2] != 3:
    #    raise ValueError("Input image must be a 3-channel (BGR) color image.")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
