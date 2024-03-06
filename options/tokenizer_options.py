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


class TokenizerOptions:
    """
    Abstract class defining the options used for models
    """
    device: Union[str, Devices]
    padding_side: str
    return_tensors: str = "pt"

    def __init__(self, device: Devices, padding_side: str):
        """
        Initializes the options class with the given device
             :param device:  used for generation
        """
        self.device = device
        if padding_side not in ['left', 'right']:
            raise ValueError("padding_side must be either 'left' or 'right'")
        self.padding_side = padding_side

