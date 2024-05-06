from transformers import AutoTokenizer, PreTrainedTokenizer
from typing import Union

from sdk.options.options import Devices


class Tokenizer:

    """
    Abstract base class for all tokenizers.

    Attributes:
        pipeline (PreTrainedTokenizer): The instantiated tokenizer.
        device (Union[str, Devices]): The device on which the tokenizer
            operates.
        tokenizer_name (str): The name of the tokenizer.
        tokenizer_path (str): The path of the tokenizer.
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
        __init__ Initializes the TokenizerObject class with the given parameters.

        :param tokenizer_name: The name of the tokenizer.
        :param tokenizer_path: The path of the tokenizer.
        :param device: The device on which
                the tokenizer operates.
        :param **kwargs: Additional keyword arguments to pass.
        """
        self.tokenizer_name = tokenizer_name
        self.tokenizer_path = tokenizer_path
        self.device = device
        self.create_tokenizer(**kwargs)

    def create_tokenizer(self, **kwargs):
        """
        Creates the tokenizer to use.

        :param **kwargs: Additional keyword arguments to pass.
        """
        self.pipeline = AutoTokenizer.from_pretrained(
            self.tokenizer_path,
            **kwargs
        )

    def encode(self, prompt: str, **kwargs):
        """
        encode Tokenizes the given prompt and prepares it for model input.

        :param prompt: The prompt to encode.
        :param **kwargs: Additional options to pass to the encode method.

        :return: The tokenized and formatted input tensor.
        """
        return self.pipeline.encode(
            prompt,
            **kwargs,
        ).to(device=(self.device if isinstance(
            self.device, str) else (
                self.device.value)))
