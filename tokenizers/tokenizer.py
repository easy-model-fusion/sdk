from abc import abstractmethod
from typing import Optional

from sdk.options.tokenizer_options import TokenizerOptions


class Tokenizer:
    """
    Abstract base class for all tokenizers
    """
    model_max_length: int

    def __init__(self, model_max_length):
        self.model_max_length = model_max_length

