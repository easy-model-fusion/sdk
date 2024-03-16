from typing import Union, Optional
from sdk import Devices


class OptionsTokenizer:
    """
    Abstract class defining the options used for models
    """
    device: Union[str, Devices]
    padding_side: str = "left"
    return_tensors: Optional[str] = None

    def __init__(self, device: Union[str, Devices],
                 padding_side: Optional[str],
                 return_tensors: Optional[str]):
        """
        Initializes the options class with the given device and other options.

        Args:
            device (Union[str, Devices]): The device used for generation.
            padding_side (str): The padding side,
                either 'left' or 'right'. Defaults to 'left'.
            return_tensors (str): The return tensor format. Defaults to 'pt'.
        """
        self.device = device
        if padding_side:
            if padding_side not in ['left', 'right']:
                raise ValueError(
                    "padding_side must be either 'left' or 'right'")
        else:
            self.padding_side = padding_side
        if return_tensors:
            self.return_tensors = return_tensors
