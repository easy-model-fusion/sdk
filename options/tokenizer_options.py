from abc import ABC
from typing import Union
from enum import Enum


class Devices(Enum):
    """
    Enumeration of devices supported
    """
    GPU = "cuda"
    CPU = "cpu"
    RESET = "meta"


class TokenizerOptions(ABC):
    """
    Abstract class defining the options used for models
    """
    device: Union[str, Devices]

    def __init__(self, device: Devices):
        """
        Initializes the options class with the given device
             :param device:  used for generation
        """
        self.device = device
