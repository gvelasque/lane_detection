# laneLineDetection/lane_detection_lib/image/channel.py
# Channel extraction functions

from ..common import Enum, ImageType


class ChannelType(Enum):
    """Enum for BGR channels"""
    BLUE = 0
    GREEN = 1
    RED = 2


def extract_channel(img: ImageType, channel_type: ChannelType) -> ImageType:
    """Extract a specific channel from a BGR image."""
    # if img.ndim != 3 or img.shape[2] not in [1, 3]:
    #    raise ValueError("Input image must be a 3-channel (BGR) image.")
    if not isinstance(channel_type, ChannelType):
        raise ValueError("Channel must be an instance of ChannelType Enum.")

    return img[:, :, channel_type.value]
