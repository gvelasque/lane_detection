# laneLineDetection/lane_detection_lib/image/edge_detection.py
# Edge detection functions

from ..common import cv2, np, ImageType


def validate_threshold(threshold_lower: int, threshold_higher: int) -> None:
    if not (isinstance(threshold_lower, int)
            and isinstance(threshold_higher, int)):
        raise ValueError("Thresholds must be integers.")

    if (threshold_lower < 0 or threshold_higher < 0
            or threshold_lower >= threshold_higher):
        raise ValueError(
            "Threshold values must be positive, and lower < higher.")


def apply_canny_edge_detection(img: ImageType, threshold_lower: int,
                               threshold_higher: int) -> ImageType:
    """Perform Canny edge detection on the input image."""
    validate_threshold(threshold_lower, threshold_higher)

    return cv2.Canny(img, threshold_lower, threshold_higher)


def detect_hough_lines(img: ImageType) -> ImageType:
    # Parameters for the Hough Line Transform
    rho = 1  # Distance resolution in pixels
    theta = np.pi / 180  # Angular resolution in radians (1 degree)
    threshold = 30  # Minimum number of intersections
    min_line_length = 50  # Minimum length of a line (pixels)
    max_line_gap = 15  # Maximum gap between lines (pixels)

    return cv2.HoughLinesP(
        img, rho, theta, threshold, np.array([]),
        minLineLength=min_line_length, maxLineGap=max_line_gap
    )
