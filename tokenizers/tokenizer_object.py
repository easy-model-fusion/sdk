from abc import abstractmethod
import torch
from transformers import AutoTokenizer

class TokenizerObject:
    """
    Abstract base class for all tokenizers
    """
    tokenizer: AutoTokenizer
    model_name: str
    model_path: str
    user_input: str
    chatbot_output: str
    conversation_active: bool = False

    def __init__(self, model_name, model_path: str):
        """
        Initializes the model with the given name
        :param model_name: The name of the model
        :param model_path: The path of the model
        """
        self.model_name = model_name
        self.model_path = model_path
        self.create_tokenizer()

    def create_tokenizer(self):
        """
        Creates the tokenizer to use
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            padding_side='left'
            # model_max_length=tokenizer_options.model_max_length,
            # Add more arguments ?
        )

    def prepare_input(self, prompt: str, ):
        return self.tokenizer.encode(prompt, return_tensors="pt").to('cuda')

    def decode_model_output(self, outputs: torch.Tensor):
        return self.tokenizer.decode(outputs[0])
