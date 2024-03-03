from typing import Optional
import torch
from transformers import PreTrainedTokenizerFast, Conversation
from sdk import ModelsTextConversation
from sdk.options.fast_pretrained_tokenizer_options import FastPreTrainedTokenizerOptions
from sdk.tokenizers.tokenizer_object import TokenizerObject


class PreTrained(TokenizerObject):
    """
    Abstract base class for all PreTrainedFast Tokenizers
    """
    tokenizer: PreTrainedTokenizerFast
    chat_bot: Conversation
    chat_history_ids: torch.Tensor
