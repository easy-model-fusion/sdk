from abc import abstractmethod


class Model:
    """
    Abstract base class for all models
    """
    model_name: str
    model_path: str

    def __init__(self, model_name, model_path: str):
        """
        Initializes the model with the given name
        :param model_name: The name of the model
        :param model_path: The path of the model
        """
        self.model_name = model_name
        self.model_path = model_path

    @abstractmethod
    def load_model(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def unload_model(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def generate_prompt(self, prompt: str, **kwargs):
        raise NotImplementedError
