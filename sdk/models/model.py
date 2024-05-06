from abc import abstractmethod
from typing import Union

from sdk.options import Devices


class Model:
    """
    Model Abstract base class for all models.

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
        __init__ Initializes the model with the given name.


        :param model_name: The name of the model.
        :param model_path:  The path of the model.
        :param device: Defines which device the model must be on.
        :param single_file: Defines whether model is single file or not.

        :return: Returns an instance of model
        """
        self.model_name = model_name
        self.model_path = model_path
        self.device = device
        self.loaded = False
        self.single_file = single_file

    @abstractmethod
    def load_model(self) -> bool:
        """
        load_model Loads the model on the given device.

        :return: True if the model is successfully loaded.
        """
        raise NotImplementedError

    @abstractmethod
    def unload_model(self) -> bool:
        """
        unload_model unloads the model.

        :returns: True if the model is successfully unloaded.
        """
        raise NotImplementedError

    @abstractmethod
    def generate_prompt(self, prompt: str, **kwargs):
        """
        generate_prompt Generates the prompt with the given option.

        :param prompt: The prompt used to generate
        :param kwargs: Additional parameters for generating the prompt.
        """
        raise NotImplementedError
