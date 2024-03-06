from abc import abstractmethod
import torch
from transformers import AutoTokenizer

from sdk.options.tokenizer_options import TokenizerOptions


class TokenizerObject:
    """
    Abstract base class for all tokenizers
    """
    tokenizer: AutoTokenizer
    options: TokenizerOptions
    model_name: str
    model_path: str
    user_input: str
    chatbot_output: str
    conversation_active: bool = False

    def __init__(self, model_name: str,
                 model_path: str,
                 options: TokenizerOptions):
        """
        Initializes the model with the given name
        :param model_name: The name of the model
        :param model_path: The path of the model
        """
        self.model_name = model_name
        self.model_path = model_path
        self.options = options
        self.create_tokenizer()

    def create_tokenizer(self):
        """
        Creates the tokenizer to use
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            padding_side=self.options.padding_side
        )

    def prepare_input(self, prompt: str, ):
        return self.tokenizer.encode(
            prompt,
            return_tensors=
            self.options.return_tensors).to(self.options.device)

    def decode_model_output(self, outputs: torch.Tensor):
        return self.tokenizer.decode(outputs[0])
