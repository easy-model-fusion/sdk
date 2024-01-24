from abc import ABC, abstractmethod
from ..options.options import Options


class Models(ABC):
    model_mame: str

    @abstractmethod
    def load_model(self, option: Options):
        raise NotImplementedError

    @abstractmethod
    def unload_model(self):
        raise NotImplementedError

    @abstractmethod
    def generate_prompt(self, option: Options):
        raise NotImplementedError
