import torch
from transformers import AutoTokenizer

from sdk.options.options_tokenizer import OptionsTokenizer


class TokenizerObject:
    """
    Abstract base class for all tokenizers
    """
    pipeline: AutoTokenizer
    options: OptionsTokenizer
    model_name: str
    tokenizer_path: str
    user_input: str
    chatbot_output: str
    conversation_active: bool = False

    def __init__(self, model_name: str,
                 model_path: str,
                 options: OptionsTokenizer):
        """
        Initializes the TokenizerObject class with the given parameters.

        Args:
            model_name (str): The name of the model.
            model_path (str): The path of the model.
            options (OptionsTokenizer): The options for the tokenizer.
        """
        self.model_name = model_name
        self.tokenizer_path = model_path
        self.options = options
        self.create_tokenizer()

    def create_tokenizer(self):
        """
        Creates the tokenizer to use
        """
        self.pipeline = AutoTokenizer.from_pretrained(
            self.tokenizer_path,
            trust_remote_code=True,
            padding_side=self.options.padding_side
        )

    def prepare_input(self, prompt: str, history: str):
        """
        Tokenizes the given prompt and prepares it for model input.

        Args:
            history: chat history
            prompt (str): The input prompt.

        Returns:
            torch.Tensor: The tokenized and formatted input tensor.
        """
        return self.pipeline.encode(
            prompt,
            history,
            return_tensors='pt',
            truncation=True
        ).to(torch.device("cuda"))

    def decode_model_output(self, outputs: torch.Tensor):
        """
        Decodes the model output tensor to obtain the generated text.

        Args:
            outputs (torch.Tensor): The model output tensor.

        Returns:
            str: The decoded generated text.
        """
        return self.pipeline.decode(outputs[0]).strip()
