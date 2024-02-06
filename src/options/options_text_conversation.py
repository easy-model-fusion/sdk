from transformers.feature_extraction_utils import PreTrainedFeatureExtractor
from transformers.image_processing_utils import BaseImageProcessor

from src.options.options import Options, Devices
import torch
from typing import Optional, Union, List, Dict, Tuple, Any, Callable
from transformers import pipeline, PreTrainedModel, TFPreTrainedModel, PretrainedConfig, PreTrainedTokenizer, \
    PreTrainedTokenizerFast, ModelCard


class OptionsTextConversation(Options):
    """
    Options for text-Generation models
    """
    """
    Args:
       model (str or PreTrainedModel or TFPreTrainedModel, optional):
           The model that will be used by the pipeline to make predictions. This can be a model identifier or an
           actual instance of a pretrained model inheriting from `PreTrainedModel` (for PyTorch) or
           `TFPreTrainedModel` (for TensorFlow).
    
           If not provided, the default for the `task` will be loaded.
       config (str or PretrainedConfig, optional):
           The configuration that will be used by the pipeline to instantiate the model. This can be a model
           identifier or an actual pretrained model configuration inheriting from `PretrainedConfig`.
    
           If not provided, the default configuration file for the requested model will be used. That means that if
           `model` is given, its default configuration will be used. However, if `model` is not supplied, this
           `task`'s default model's config is used instead.
       tokenizer (str or PreTrainedTokenizer, optional):
           The tokenizer that will be used by the pipeline to encode data for the model. This can be a model
           identifier or an actual pretrained tokenizer inheriting from `PreTrainedTokenizer`.
    
           If not provided, the default tokenizer for the given `model` will be loaded (if it is a string). If `model`
           is not specified or not a string, then the default tokenizer for `config` is loaded (if it is a string).
           However, if `config` is also not given or not a string, then the default tokenizer for the given `task`
           will be loaded.
       feature_extractor (str or PreTrainedFeatureExtractor, optional):
           The feature extractor that will be used by the pipeline to encode data for the model. This can be a model
           identifier or an actual pretrained feature extractor inheriting from `PreTrainedFeatureExtractor`.
    
           Feature extractors are used for non-NLP models, such as Speech or Vision models as well as multi-modal
           models. Multi-modal models will also require a tokenizer to be passed.
    
           If not provided, the default feature extractor for the given `model` will be loaded (if it is a string). If
           `model` is not specified or not a string, then the default feature extractor for `config` is loaded (if it
           is a string). However, if `config` is also not given or not a string, then the default feature extractor
           for the given `task` will be loaded.
       framework (str, optional):
           The framework to use, either "pt" for PyTorch or "tf" for TensorFlow. The specified framework must be
           installed.
    
           If no framework is specified, will default to the one currently installed. If no framework is specified and
           both frameworks are installed, will default to the framework of the `model`, or to PyTorch if no model is
           provided.
       revision (str, optional, defaults to "main"):
           When passing a task name or a string model identifier: The specific model version to use. It can be a
           branch name, a tag name, or a commit id, since we use a git-based system for storing models and other
           artifacts on huggingface.co, so `revision` can be any identifier allowed by git.
       use_fast (bool, optional, defaults to True):
           Whether or not to use a Fast tokenizer if possible (a `PreTrainedTokenizerFast`).
       use_auth_token (str or bool, optional):
           The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
           when running `huggingface-cli login` (stored in `~/.huggingface`).
       device (int or str or torch.device, optional):
           Defines the device (*e.g.*, "cpu", "cuda:1", "mps", or a GPU ordinal rank like 1) on which this
           pipeline will be allocated.
       device_map (str or Dict[str, Union[int, str, torch.device]], optional):
           Sent directly as `model_kwargs` (just a simpler shortcut). When `accelerate` library is present, set
           `device_map="auto"` to compute the most optimized `device_map` automatically.
    
           Do not use `device_map` AND `device` at the same time as they will conflict.
       torch_dtype (str or torch.dtype, optional):
           Sent directly as `model_kwargs` (just a simpler shortcut) to use the available precision for this model
           (`torch.float16`, `torch.bfloat16`, ... or "auto").
       trust_remote_code (bool, optional, defaults to False):
           Whether or not to allow for custom code defined on the Hub in their own modeling, configuration,
           tokenization or even pipeline files. This option should only be set to `True` for repositories you trust
           and in which you have read the code, as it will execute code present on the Hub on your local machine.
       model_kwargs (Dict[str, Any], optional):
           Additional dictionary of keyword arguments passed along to the model's `from_pretrained(...,
           **model_kwargs)` function.
       kwargs (Dict[str, Any], optional):
           Additional keyword arguments passed along to the specific pipeline init.
        """
    device : Devices
    prompt: str = ""
    model: Union[str, PreTrainedModel, "TFPreTrainedModel"] = None
    tokenizer: Union[str, PreTrainedTokenizer, "PreTrainedTokenizerFast"] = None
    model_card: Optional[Union[str, ModelCard]] = None
    framework: Optional[str] = None
    task: str = ""
    num_workers: Optional[int] = 8
    batch_size: Optional[int] = 1
    device: Optional[Union[int, str, "torch.device"]] = -1
    arg_parser: Optional[Dict[str, Any]] = None
    torch_dtype: Optional[Union[str, torch.dtype]] = None
    binary_output: Optional[bool] = False
    min_length_for_response: Optional[int] = 32
    minimum_tokens: Optional[int] = 10

    def __init__(self,
                 device: Devices,
                 prompt : str = "",
                 model: Union[str, PreTrainedModel, "TFPreTrainedModel"] = None,
                 tokenizer: Union[str, PreTrainedTokenizer, "PreTrainedTokenizerFast"] = None,
                 model_card: Optional[Union[str, ModelCard]] = None,
                 framework: Optional[str] = None,
                 task: str = "",
                 num_workers: Optional[int] = 8,
                 batch_size: Optional[int] = 1,
                 arg_parser: Optional[Dict[str, Any]] = None,
                 torch_dtype: Optional[Union[str, torch.dtype]] = None,
                 binary_output: Optional[bool] = False,
                 min_length_for_response: Optional[int] = 32,
                 minimum_tokens: Optional[int] = 10,
                 ):
        """
        Initializes the OptionsTextGeneration.

        Args:
            device (Devices): The device to use to generate the prompt.
            prompt (str, optional): The prompt to give to the model.
            model (str or PreTrainedModel or TFPreTrainedModel, optional): The model to use for text generation.
            tokenizer (str or PreTrainedTokenizer or PreTrainedTokenizerFast, optional): The tokenizer to use for
                encoding the prompt and decoding the generated text.
            model_card (str or ModelCard, optional): The model card to provide additional information about the model.
            framework (str, optional): The framework to use for the model (e.g., "pt" for PyTorch, "tf" for TensorFlow).
            task (str, optional): The task for which the model is being used.
            num_workers (int, optional): The number of worker processes for data loading.
            batch_size (int, optional): The batch size for inference.
            arg_parser (Dict[str, Any], optional): Additional arguments to parse.
            torch_dtype (str or torch.dtype, optional): The data type to use for PyTorch tensors.
            binary_output (bool, optional): Whether the output should be binary.
            min_length_for_response (int, optional): The minimum length of the generated response.
            minimum_tokens (int, optional): The minimum number of tokens required for the generated response.
        """
        super().__init__(device)
        # Initialize additional attributes
        self.prompt = prompt
        self.task = task
        if model:
            self.model = model
        if tokenizer:
            self.tokenizer = tokenizer
        if model_card:
            self.model_card = model_card
        if num_workers:
            self.num_workers = num_workers
        if batch_size:
            self.batch_size = batch_size
        if framework:
            self.framework = framework
        if arg_parser:
            self.arg_parser = arg_parser
        if torch_dtype:
            self.torch_dtype = torch_dtype
        self.torch_dtype = torch_dtype
        if binary_output:
            self.binary_output = binary_output
        if min_length_for_response:
            self.min_length_for_response = min_length_for_response
        if minimum_tokens:
            self.minimum_tokens = minimum_tokens
