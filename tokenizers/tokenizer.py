from typing import Optional


class Tokenizer:
    """
    Abstract base class for all tokenizers
    """
    model_name: str

    def __init__(self, model_name: str):
        """
        Initializes the Tokenizer with the given name
        :param model_name: The name of the model
        """
        self.model_name = model_name
