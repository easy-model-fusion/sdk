from abc import ABC
from transformers.tokenization_utils_base import VERY_LARGE_INTEGER


class TokenizerOptions(ABC):
    """
    Abstract class defining the options used for tokenizers
    """

    model_max_length: int = VERY_LARGE_INTEGER,

    def __init__(self, model_max_length: int):
        """
        Initializes the options class with the given device
        :param model_max_length: The max length of the model
        """
        self.model_max_length = model_max_length
