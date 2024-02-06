from abc import ABC, abstractmethod
from src.options.options import Options


class Models(ABC):
    """
    Abstract base class for all models
    """
    model_name: str

    def __init__(self, model_name: str):
        """
        Initializes the model with the given name
        :param model_name: The name of the model
        """
        self.model_name = model_name

    @abstractmethod
    def load_model(self, option: Options) -> bool:
        raise NotImplementedError

    @abstractmethod
    def unload_model(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def generate_prompt(self, option: Options):
        raise NotImplementedError
