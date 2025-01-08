# lane_detection/app/config.py
import os
import logging

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Input and output image directories
INPUT_DIR = os.path.join(BASE_DIR, "data", "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "output")

# Default file paths (can be changed dynamically)
DEFAULT_IMAGE_PATH = os.path.join(INPUT_DIR, "test_route.jpeg")
DEFAULT_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "processed_images.jpeg")

# Ensure directories exist
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def setup_logging():
    """Configure the logging system."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
