from abc import abstractmethod
from typing import Union

from sdk.options import Devices


class Model:
    """
    Abstract base class for all models.

    Attributes:
        model_name (str): The name of the model.
        model_path (str): The path of the model.
        device (Union[str, Devices]): Which device the model must be on.
        loaded (bool): Indicates if the model is loaded.
        single_file (bool): Indicates if the model is a single file.
    """

    model_name: str
    model_path: str
    device: Union[str, Devices]
    loaded: bool
    single_file: bool

    def __init__(self, model_name, model_path: str,
                 device: Union[str, Devices], single_file: bool = False):
        """
        Initializes the model with the given name.

        Args:
            model_name (str): The name of the model.
            model_path (str): The path of the model.
            device (Union[str, Devices]): Which device the model must be on.
            single_file (bool, optional): Whether model is single file or not.
        """
        self.model_name = model_name
        self.model_path = model_path
        self.device = device
        self.loaded = False
        self.single_file = single_file

    @abstractmethod
    def load_model(self) -> bool:
        """
        Load this model on the given device.

        Returns:
            bool: True if the model is successfully loaded.
        """
        raise NotImplementedError

    @abstractmethod
    def unload_model(self) -> bool:
        """
        Unloads the model.

        Returns:
            bool: True if the model is successfully unloaded.
        """
        raise NotImplementedError

    @abstractmethod
    def generate_prompt(self, prompt: str, **kwargs):
        """
        Generates the prompt with the given option.

        Args:
            prompt (str): The prompt to generate.
            **kwargs: Additional parameters for generating the prompt.
        """
        raise NotImplementedError
