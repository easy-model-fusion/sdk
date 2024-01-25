from abc import ABC, abstractmethod
from src.options.options import Options


class Models(ABC):
    model_name: str

    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def load_model(self, option: Options):
        raise NotImplementedError

    @abstractmethod
    def unload_model(self):
        raise NotImplementedError

    @abstractmethod
    def generate_prompt(self, option: Options):
        raise NotImplementedError
