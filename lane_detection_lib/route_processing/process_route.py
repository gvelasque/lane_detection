# laneLineDetection/lane_detection_lib/route_processing/process_route.py
# Processing pipeline

from lane_detection_lib.common import Path, cv2, TypeAlias
from lane_detection_lib.draw.lines_detection import detect_and_draw_lanes

from lane_detection_lib.image.blur import apply_gaussian_blur
from lane_detection_lib.image.color_conversion import convert_to_rgb2grayscale
from lane_detection_lib.image.edge_detection import apply_canny_edge_detection
from lane_detection_lib.image.io import load_image, save_image
from lane_detection_lib.image.resize import resize_by_factor
from lane_detection_lib.image.roi import apply_roi_mask, MaskType


def process_route(image_path: str, output_path: str = None) -> TypeAlias:
    """Load, process, and display a route detection image."""
    image_file = Path(image_path)

    if not image_file.is_file():
        raise FileNotFoundError(
            f"Image file '{image_path}' doesn't exist. Please check the path.")

    # Load image
    image = load_image(str(image_file))
    # Resize the image
    resized_image = resize_by_factor(image, factor_x=0.5, factor_y=0.5)
    # Convert the image to grayscale
    gray_image = convert_to_rgb2grayscale(resized_image)
    # Apply Gaussian blur
    blurred_image = apply_gaussian_blur(gray_image, (5, 5), 0)
    # Apply Canny edge detection
    edges = apply_canny_edge_detection(blurred_image, 50, 175)
    # Get the mask for the region of interest
    roi_mask = apply_roi_mask(edges, MaskType.triangle)
    # Apply the mask to the edges image
    masked_edges = cv2.bitwise_and(edges, roi_mask)
    # Detect and draw lines
    final_image = detect_and_draw_lanes(resized_image, masked_edges)

    # Save image
    if output_path is not None:
        save_image(final_image, output_path)

    return final_image
