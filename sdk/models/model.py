from abc import abstractmethod
from sdk.options import Options
from typing import Optional


class Model:
    """
    Abstract base class for all models
    """
    model_name: str
    model_path: str
    single_file: bool

    def __init__(self, model_name, model_path: str, single_file: bool = False):
        """
        Initializes the model with the given name
        :param model_name: The name of the model
        :param model_path: The path of the model
        :param single_file: Whether model is single file or not
        """
        self.model_name = model_name
        self.model_path = model_path
        self.single_file = single_file

    @abstractmethod
    def load_model(self, option: Options) -> bool:
        raise NotImplementedError

    @abstractmethod
    def unload_model(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def generate_prompt(self, prompt: Optional[str],
                        option: Options, **kwargs):
        raise NotImplementedError
