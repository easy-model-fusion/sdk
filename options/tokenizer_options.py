from abc import ABC
from typing import Union, Optional
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
    padding_side: str = "left"
    return_tensors: str = "pt"
    kwargs: Optional[dict] = None

    def __init__(self, device: str,
                 padding_side: str = "left",
                 return_tensors: str = "pt",
                 **kwargs
                 ):
        """
        Initializes the options class with the given device
             :param device:  used for generation
        """
        self.device = device
        if padding_side not in ['left', 'right']:
            raise ValueError("padding_side must be either 'left' or 'right'")
        self.padding_side = padding_side
        self.return_tensors = return_tensors
        self.kwargs = kwargs

