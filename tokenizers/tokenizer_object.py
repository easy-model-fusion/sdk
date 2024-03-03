from abc import abstractmethod
from typing import Optional
from transformers import AutoTokenizer
from sdk import ModelsTextConversation
from sdk.options.tokenizer_options import TokenizerOptions


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

    @abstractmethod
    def start_conversation(self, prompt: Optional[str],
                           user_input: str,
                           model: ModelsTextConversation,
                           option: TokenizerOptions) -> bool:
        raise NotImplementedError
