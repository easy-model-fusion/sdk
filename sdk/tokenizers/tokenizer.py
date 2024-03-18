import torch
from transformers import AutoTokenizer

from sdk.options.options_tokenizer import OptionsTokenizer


class Tokenizer:
    """
    Abstract base class for all tokenizers
    """
    pipeline: AutoTokenizer
    options: OptionsTokenizer
    model_name: str
    tokenizer_name: str
    tokenizer_path: str
    user_input: str
    chatbot_output: str
    conversation_active: bool = False
    return_tensors = 'pt'

    def __init__(self, model_name: str,
                 tokenizer_path: str,
                 tokenizer_name: str,
                 options: OptionsTokenizer):
        """
        Initializes the TokenizerObject class with the given parameters.

        Args:
            model_name (str): The name of
                the model associated with the tokenizer .
            tokenizer_name (str): The name of the tokenizer
            tokenizer_path (str): The path of the tokenizer
            options (OptionsTokenizer): The options for the tokenizer.
        """
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.tokenizer_path = tokenizer_path
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
            history,
            prompt,
            return_tensors=self.return_tensors,
            truncation=True
        ).to(self.options.device)

    def decode_model_output(self, outputs: torch.Tensor):
        """
        Decodes the model output tensor to obtain the generated text.

        Args:
            outputs (torch.Tensor): The model output tensor.

        Returns:
            str: The decoded generated text.
        """
        return self.pipeline.decode(outputs[0]).strip()
