# laneLineDetection/lane_detection_lib/common.py
# Shared type aliases and imports

import logging
from pathlib import Path
from enum import Enum
from typing import TypeAlias

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Type Alias
ImageType: TypeAlias = np.ndarray
