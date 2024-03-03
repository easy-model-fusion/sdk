from typing import Optional
import torch
from transformers import PreTrainedTokenizerFast, Conversation
from sdk import ModelsTextConversation
from sdk.options.fast_pretrained_tokenizer_options import FastPreTrainedTokenizerOptions
from sdk.tokenizers.tokenizer_object import TokenizerObject


class PreTrainedFast(TokenizerObject):
    """
    Abstract base class for all PreTrainedFast Tokenizers
    """
    tokenizer: PreTrainedTokenizerFast
    chat_bot: Conversation
    chat_history_ids: torch.Tensor

    def __init__(self, model_name: str, model_path: str, tokenizer_options: FastPreTrainedTokenizerOptions):
        """
        Initializes the ModelsTextToImage class
        :param model_name: The name of the model
        :param model_path: The path of the model
        :param tokenizer_options: Options for the tokenizer
        """
        super().__init__(model_name, model_path)
        self.create_tokenizer(tokenizer_options)
        self.chat_history_ids = torch.tensor([], dtype=torch.long)  # Initialize chat_history_ids here

    def create_tokenizer(self, tokenizer_options: FastPreTrainedTokenizerOptions):
        """
        Creates the pipeline to load on the device
        """
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            padding_side=tokenizer_options.padding_side,
            model_max_length=tokenizer_options.model_max_length,
            # Add more arguments ?
        )
