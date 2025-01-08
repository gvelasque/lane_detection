# lane_detection/app/main.py

import logging

from lane_detection_lib.image.io import display_image_plt
from lane_detection_lib.route_processing.process_route import process_route
from app.config import DEFAULT_IMAGE_PATH, DEFAULT_OUTPUT_PATH

from config import setup_logging


def setup():
    """Set up the application configuration."""
    setup_logging()


if __name__ == "__main__":
    setup()  # Set configuration of project

    logging.info("Starting route detection process...")

    IMAGE_PATH = DEFAULT_IMAGE_PATH
    OUTPUT_PATH = DEFAULT_OUTPUT_PATH

    try:
        processed_route = process_route(IMAGE_PATH,
                                        output_path=OUTPUT_PATH)
        display_image_plt(processed_route, "Processed Route Image")
    except Exception as e:
        logging.exception(f"Unexpected error: {e}")
    finally:
        logging.info("Processing complete.")
