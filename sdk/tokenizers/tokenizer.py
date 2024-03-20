from transformers import AutoTokenizer, PreTrainedTokenizer
from typing import Union

from sdk.options.options import Devices


class Tokenizer:
    """
    Abstract base class for all tokenizers
    """
    pipeline: PreTrainedTokenizer
    device: Union[str, Devices]
    tokenizer_name: str
    tokenizer_path: str

    def __init__(self,
                 tokenizer_name: str,
                 tokenizer_path: str,
                 device: Union[str, Devices],
                 **kwargs):
        """
        Initializes the TokenizerObject class with the given parameters.

        Args:
            tokenizer_name (str): The name of the tokenizer
            tokenizer_path (str): The path of the tokenizer
        """
        self.tokenizer_name = tokenizer_name
        self.tokenizer_path = tokenizer_path
        self.device = device
        self.create_tokenizer(**kwargs)

    def create_tokenizer(self, **kwargs):
        """
        Creates the tokenizer to use
        """
        self.pipeline = AutoTokenizer.from_pretrained(
            self.tokenizer_path,
            **kwargs
        )

    def encode(self, prompt: str, **kwargs):
        """
        Tokenizes the given prompt and prepares it for model input.

        :param prompt: (str): The prompt to encode
        :param kwargs: (dict): Additional options to pass to the encode method

        Returns:
            torch.Tensor: The tokenized and formatted input tensor.
        """
        return self.pipeline.encode(
            prompt,
            **kwargs,
        ).to(device=(self.device if isinstance(
            self.device, str) else (
                self.device.value)))