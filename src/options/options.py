from abc import ABC
from enum import Enum


class Devices(Enum):
    """
    Enumeration of devices supported
    """
    GPU = "cuda"
    CPU = "cpu"
    RESET = "meta"


class Options(ABC):
    """
    Abstract class defining the options used for models
    """
    device: Devices

    def __init__(self, device: Devices):
        """
        Initializes the options class with the given device
        :param device: The device to use generate prompt
        """
        self.device = device
