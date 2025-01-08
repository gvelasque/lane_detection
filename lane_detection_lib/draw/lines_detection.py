# laneLineDetection/lane_detection_lib/draw/lines_detection.py
from typing import Optional

from lane_detection_lib.image.color_conversion import convert_bgr2rgb
from lane_detection_lib.image.edge_detection import detect_hough_lines
from ..common import cv2, np, ImageType
from ..image.io import get_image_dimensions


# ----------------- Helper Functions -----------------

def get_slope_line(line: np.ndarray) -> float:
    """Calculate the slope of the line"""
    if not isinstance(line, np.ndarray) or line.shape != (1, 4):
        raise ValueError("Line must be a NumPy array with shape (1, 4).")


    # Unpack the line coordinates and cast
    x1, y1, x2, y2 = (int(coord) for coord in line[0])
    h_dist = x2 - x1
    v_dist = y2 - y1

    # Return slope, handling vertical lines
    return v_dist / h_dist if h_dist != 0 else float('inf')


def extract_line_coordinates(line: np.ndarray) -> tuple[list[int], list[int]]:
    """Extract x and y coordinates from a single line."""
    if not isinstance(line, np.ndarray) or line.shape != (4,):
        raise ValueError("Each line must be a numpy array with "
                         "exactly four elements (x1, y1, x2, y2).")

    if not np.issubdtype(line.dtype, np.integer):
        raise TypeError("Line coordinates must be integers.")

    x_start, y_start, x_end, y_end = line
    return [x_start, x_end], [y_start, y_end]


def get_all_line_coordinates(
        lines: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extracts all x and y coordinates from multiple detected lines."""
    if not isinstance(lines, np.ndarray):
        raise TypeError("Lines must be a numpy array.")

    all_x_coords, all_y_coords = [], []

    for line in lines:
        x_coords, y_coords = extract_line_coordinates(line)
        all_x_coords.extend(x_coords)
        all_y_coords.extend(y_coords)

    return np.array(all_x_coords), np.array(all_y_coords)


# ----------------- Line Processing Functions -----------------

def separate_lines(
        image: ImageType, lines: np.ndarray,
        slope_threshold: float = 0.50) -> tuple[np.ndarray, np.ndarray]:
    """ Separates detected lines into left and right lane lines based on slope."""
    left_lines = []
    right_lines = []
    height, width = get_image_dimensions(image)
    center_x = width // 2  # Center of the image

    # Return empty lists if no lines are detected
    if lines is None or len(lines) == 0:
        return np.empty((0, 4), dtype=int), np.empty((0, 4), dtype=int)

    if not isinstance(lines, (list, np.ndarray)):
        raise TypeError("Lines must be a list or a NumPy array.")

    for line in lines:
        # Unpack the line coordinates and cast
        x1, y1, x2, y2 = (int(coord) for coord in line[0])
        slope = get_slope_line(line)

        # Ignore near-horizontal lines
        if abs(slope) < slope_threshold:
            continue

        # Separate left and right lines
        if slope < 0 and x1 < center_x and x2 < center_x:
            left_lines.append((x1, y1, x2, y2))
        elif slope > 0 and x1 > center_x and x2 > center_x:
            right_lines.append((x1, y1, x2, y2))

    return np.array(left_lines), np.array(right_lines)


def fit_detected_line(image: ImageType,
                      detected_lines: np.ndarray) -> tuple[int, ...] | None:
    """Fits a single line to detected lane lines using linear regression."""
    if detected_lines is None or not isinstance(
            detected_lines, np.ndarray) or len(detected_lines) == 0:
        return None  # No lines detected, return None.

    height, width = get_image_dimensions(image)

    # Extract x and y coordinates
    x_coords, y_coords = get_all_line_coordinates(detected_lines)

    if len(x_coords) < 2 or len(y_coords) < 2:
        raise None

    # Fit a line to the points (y = mx + b)
    poly = np.polyfit(y_coords, x_coords, 1)  # Reverse xy (Vertical lines)
    slope, intercept = poly

    # Define start and end points of the fitted line
    y_start = height  # Start from the bottom
    y_end = int(height * 0.65)  # Extend to 65% of the image height
    x_start = int(slope * y_start + intercept)
    x_end = int(slope * y_end + intercept)

    # Validate calculated coordinates
    if not (0 <= x_start < width and 0 <= x_end < width):
        raise ValueError("Fitted line coordinates are out of image bounds.")

    return x_start, y_start, x_end, y_end


# ----------------- Drawing Functions -----------------


def draw_lines(img: ImageType, lines: np.ndarray) -> ImageType:
    """Draw lines on a blank image of the same shape as the input image."""

    # Create a blank image to draw lines
    line_image = np.zeros_like(img, dtype=np.uint8)

    # No lines to draw, return blank image
    if lines is None or len(lines) == 0:
        return line_image

    if not isinstance(lines, (list, np.ndarray)):
        raise TypeError("Lines must be a list or a NumPy array.")

    for line in lines:
        if not isinstance(line[0], (list, np.ndarray)) or len(line[0]) != 4:
            raise ValueError("Each line must be a list, tuple, "
                             "or array of 4 integers (x1, y1, x2, y2).")

        # Unpack the line coordinates and cast
        x1, y1, x2, y2 = (int(coord) for coord in line[0])

        if not all(isinstance(int(coord), int) for coord in (x1, y1, x2, y2)):
            raise ValueError("Line coordinates must be integers.")

        # Draw each line on the blank image
        cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)
    return line_image


def draw_lane_lines(image: ImageType, left_line: Optional[tuple[int, ...]],
                    right_line: Optional[tuple[int, ...]]) -> ImageType:
    """Draws left and right lane lines on a blank image."""
    # Create a blank image to draw lines
    lane_image = np.zeros_like(image)

    # Draw the left lane line
    if left_line is not None:
        cv2.line(lane_image, (left_line[0], left_line[1]),
                 (left_line[2], left_line[3]), (255, 0, 0), 10)

    # Draw the right lane line
    if right_line is not None:
        cv2.line(lane_image, (right_line[0], right_line[1]),
                 (right_line[2], right_line[3]), (0, 255, 0), 10)

    return lane_image


# ----------------- Main Lane Detection Pipeline -----------------
def detect_and_draw_lanes(image: ImageType, edge_img: ImageType) -> ImageType:
    """Detects and draws lane lines on the given image."""
    # Detect lines
    lines = detect_hough_lines(edge_img)

    # Separate left and right lines
    left_lines, right_lines = separate_lines(image, lines, 0.4)

    # Fit a single line for each side
    left_line = fit_detected_line(image, left_lines)
    right_line = fit_detected_line(image, right_lines)

    # Draw the detected lines
    # line_image = draw_lines(image, lines)
    lane_image = draw_lane_lines(image, left_line, right_line)

    # Overlay the lines on the original image
    final_image = cv2.addWeighted(image, 0.8, lane_image, 1, 1)

    # Convert to RGB for visualization
    return convert_bgr2rgb(final_image)
