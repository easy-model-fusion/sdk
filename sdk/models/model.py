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

    def __init__(self, model_name, model_path: str,
                 device: Union[str, Devices]):
        """
        Initializes the model with the given name

        Args:
            model_name (str): The name of the model
            model_path (str): The path of the model
            device (Union[str, Devices]): Which device the model must be on
        """
        self.model_name = model_name
        self.model_path = model_path
        self.device = device
        self.loaded = False

    @abstractmethod
    def load_model(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def unload_model(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def generate_prompt(self, prompt: str, **kwargs):
        raise NotImplementedError
