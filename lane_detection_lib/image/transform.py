# lane_detection/lane_detection_lib/image/transform.py
# Rotation and perspective transformations

from ..common import cv2, np, Enum, ImageType


class FlipType(Enum):
    """Enum for Flip Types."""
    HORIZONTAL_FLIP = 1
    VERTICAL_FLIP = 0
    BOTH_FLIP = -1


def flip_image(img: ImageType, flip_type: FlipType) -> ImageType:
    """Flip the image horizontally, vertically, or both."""

    if not isinstance(flip_type, FlipType):
        raise ValueError("flip_type must be an instance of FlipType Enum.")

    return cv2.flip(img, flip_type.value)


class RotateType(Enum):
    """Enum for Rotate Types."""
    ROTATE_90 = 90
    ROTATE_180 = 180
    ROTATE_270 = 270


def compute_new_bounding_box(rotation_matrix: ImageType, width: int,
                             height: int) -> tuple:
    """Compute the new bounding box dimensions after rotation."""

    if not isinstance(
            rotation_matrix, np.ndarray) or rotation_matrix.shape != (2, 3):
        raise ValueError(
            "rotation_matrix must be a NumPy array of shape (2,3).")

    if not isinstance(width, int) or width <= 0:
        raise ValueError("width must be a positive integer.")

    if not isinstance(height, int) or height <= 0:
        raise ValueError("height must be a positive integer.")

    # Extract rotation components (cosine and sine of the rotation angle
    cos_val = abs(rotation_matrix[0, 0])
    sin_val = abs(rotation_matrix[0, 1])

    # Compute new bounding box dimensions
    new_width = int((height * sin_val) + (width * cos_val))
    new_height = int((height * cos_val) + (width * sin_val))

    return new_width, new_height


def rotate_center(img: ImageType, rotate_type: RotateType,
                  clockwise: bool = True, scale: float = 1.0) -> ImageType:
    """Rotate the image around its center by a specified angle."""
    validate_rotation_angle(clockwise)

    if not isinstance(rotate_type, RotateType):
        raise ValueError("rotate_type must be an instance of RotateType Enum.")

    if not isinstance(scale, (int, float)) or scale <= 0:
        raise ValueError("Scale must be a positive number.")

    # Get image dimensions
    (height, width) = img.shape[:2]
    center = (width // 2, height // 2)

    # Rotate the image
    angle = -rotate_type.value if clockwise else rotate_type.value

    # Get rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)

    # Compute new bounding box
    new_width, new_height = compute_new_bounding_box(
        rotation_matrix, width, height)

    # Adjust the rotation matrix to account for translation
    rotation_matrix[0, 2] += (new_width / 2) - center[0]
    rotation_matrix[1, 2] += (new_height / 2) - center[1]

    # Rotate image
    return cv2.warpAffine(img, rotation_matrix, (new_width, new_height))


def warp_perspective(img: ImageType, src_points: np.ndarray,
                     dst_points: np.ndarray) -> ImageType:
    """Apply perspective transformation to the image."""
    if not isinstance(
            src_points, np.ndarray) or not isinstance(dst_points, np.ndarray):
        raise TypeError("src_points and dst_points must be NumPy arrays.")

    if src_points.shape != (4, 2) or dst_points.shape != (4, 2):
        raise ValueError("src_points and dst_points must have shape (4,2).")

    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    return cv2.warpPerspective(img, matrix, (img.shape[1], img.shape[0]))


def apply_affine_transform(img: ImageType, matrix: np.ndarray,
                           output_size: tuple[int, int]) -> ImageType:
    """Apply an affine transformation to the image."""
    if not isinstance(matrix, np.ndarray) or matrix.shape != (2, 3):
        raise ValueError(
            "Transformation matrix must be a NumPy array of shape (2,3).")

    if not isinstance(output_size, tuple) or len(output_size) != 2:
        raise ValueError("output_size must be a tuple of (width, height).")

    if not all(isinstance(x, int) and x > 0 for x in output_size):
        raise ValueError("output_size values must be positive integers.")

    return cv2.warpAffine(img, matrix, output_size)


def validate_rotation_angle(clockwise: bool) -> None:
    """Validate the rotation angle"""
    if not isinstance(clockwise, bool):
        raise ValueError("Rotation direction must be a boolean value.")
    # TODO: Implement function
    pass
