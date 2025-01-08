# lane_detection/lane_detection_lib/image/io.py
# Load, save, and display images

from ..common import Path, logging, cv2, np, plt, ImageType


def validate_image(img: ImageType, img_path: str) -> None:
    """Validate if the input is a proper OpenCV image."""
    if img is None:
        raise FileNotFoundError(f"Could not load image at {img_path}")

    if not isinstance(img, np.ndarray):
        raise TypeError("Input image must be a NumPy ndarray.")

    if img.ndim not in [2, 3]:
        raise ValueError(
            "Invalid image dimensions. Must be 2D (grayscale) or 3D (color).")

    if img.size == 0:
        raise ValueError("Input image is empty.")

    if img.dtype != np.uint8:
        raise TypeError("Image must be of type np.uint8.")


def load_image(img_path: str) -> ImageType:
    """Load an image from a file path."""
    if img_path is None or not isinstance(img_path, str):
        raise ValueError("Input image path must be a non-empty string.")

    img_file = Path(img_path)

    # Check if file exists
    if not img_file.is_file():
        raise FileNotFoundError(f"Error: Image not found at {img_path}")

    # Load the image
    image = cv2.imread(img_path)

    # Validate image
    validate_image(image, img_path)

    return image


def display_image_cv2(img: ImageType, window_name: str = "Image") -> None:
    """Display the image using cv2."""
    cv2.namedWindow(window_name, cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(window_name, img.shape[1], img.shape[0])
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def display_image_plt(image: ImageType, window_name: str = "Image") -> None:
    """Displays an image using Matplotlib."""
    plt.imshow(image)
    plt.axis('off')
    plt.title(window_name)
    plt.show()


def save_image(image: ImageType, output_path: str) -> bool:
    """Save an image to a file path."""
    validate_image(image, output_path)

    # Ensure the output directory exists
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(output_path, image)

    if success:
        logging.info(f"Processed image saved at {output_path}")
    else:
        logging.info(f"Failed to save image at {output_path}")

    return success


def get_aspect_ratio(image: ImageType) -> float:
    """Return aspect ratio of an image."""
    if not isinstance(image, ImageType):
        raise TypeError("Input must be an instance of ImageType.")

    return image.shape[1] / image.shape[0]


def get_image_dimensions(image: ImageType) -> tuple[int, int]:
    """Return the height and width of an image."""
    if not isinstance(image, ImageType):
        raise TypeError("Input must be an instance of ImageType.")

    return image.shape[:2]
