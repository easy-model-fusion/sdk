from abc import abstractmethod
from typing import Union

from sdk.options import Devices


class Model:
    """
    Abstract base class for all models
    """
    model_name: str
    model_path: str
    device: Union[str, Devices]
    loaded: bool
    single_file: bool

    def __init__(self, model_name, model_path: str,
                 device: Union[str, Devices], single_file: bool = False):
        """
        Initializes the model with the given name

        Args:
            model_name (str): The name of the model
            model_path (str): The path of the model
            device (Union[str, Devices]): Which device the model must be on
            :param single_file: Whether model is single file or not
        """
        self.model_name = model_name
        self.model_path = model_path
        self.device = device
        self.loaded = False
        self.single_file = single_file

    @abstractmethod
    def load_model(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def unload_model(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def generate_prompt(self, prompt: str, **kwargs):
        raise NotImplementedError
