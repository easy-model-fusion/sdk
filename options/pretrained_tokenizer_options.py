import tokenizers
from transformers.tokenization_utils_base import VERY_LARGE_INTEGER
from typing import Optional, Union, Dict, Any, List

from sdk import Devices
from sdk.options.tokenizer_options import TokenizerOptions


class PreTrainedTokenizerOptions(TokenizerOptions):
    """
    Options for PreTrained Tokenizers
    """
    device: Union[str, Devices] = -1,
    model_max_length: int = VERY_LARGE_INTEGER,
    padding_side: Optional[str] = None,
    truncation_side: Optional[str] = None,
    chat_template: Optional[str] = None,
    model_input_names: Optional[List[str]] = None,
    bos_token: Optional[Union[str, tokenizers.AddedToken]] = None,
    eos_token: Optional[Union[str, tokenizers.AddedToken]] = None,
    unk_token: Optional[Union[str, tokenizers.AddedToken]] = None,
    sep_token: Optional[Union[str, tokenizers.AddedToken]] = None,
    pad_token: Optional[Union[str, tokenizers.AddedToken]] = None,
    cls_token: Optional[Union[str, tokenizers.AddedToken]] = None,
    mask_token: Optional[Union[str, tokenizers.AddedToken]] = None,
    additional_special_tokens: Optional[List[Union[str, tokenizers.AddedToken]]] = None,
    clean_up_tokenization_spaces: bool = True,
    split_special_tokens: bool = False,

    def __init__(self,
                 device: Union[str, Devices] = -1,
                 model_max_length: int = VERY_LARGE_INTEGER,
                 padding_side: Optional[str] = None,
                 truncation_side: Optional[str] = None,
                 chat_template: Optional[str] = None,
                 model_input_names: Optional[List[str]] = None,
                 bos_token: Optional[Union[str, tokenizers.AddedToken]] = None,
                 eos_token: Optional[Union[str, tokenizers.AddedToken]] = None,
                 unk_token: Optional[Union[str, tokenizers.AddedToken]] = None,
                 sep_token: Optional[Union[str, tokenizers.AddedToken]] = None,
                 pad_token: Optional[Union[str, tokenizers.AddedToken]] = None,
                 cls_token: Optional[Union[str, tokenizers.AddedToken]] = None,
                 mask_token: Optional[Union[str, tokenizers.AddedToken]] = None,
                 additional_special_tokens: Optional[List[Union[str, tokenizers.AddedToken]]] = None,
                 clean_up_tokenization_spaces: bool = True,
                 split_special_tokens: bool = False,
                 ):
        # Initialize additional attributes
        super().__init__(device)
        self.model_max_length = model_max_length
        if padding_side:
            self.padding_side = padding_side
        if truncation_side:
            self.truncation_side = truncation_side
        if chat_template:
            self.chat_template = chat_template
        if model_input_names:
            self.model_input_names = model_input_names
        if bos_token:
            self.bos_token = bos_token
        if eos_token:
            self.eos_token = eos_token
        if unk_token:
            self.unk_token = unk_token
        if sep_token:
            self.sep_token = sep_token
        if pad_token:
            self.pad_token = pad_token
        if cls_token:
            self.cls_token = cls_token
        if mask_token:
            self.mask_token = mask_token
        if additional_special_tokens:
            self.additional_special_tokens = additional_special_tokens
        self.clean_up_tokenization_spaces = clean_up_tokenization_spaces
        self.split_special_tokens = split_special_tokens
