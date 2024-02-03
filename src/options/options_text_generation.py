from transformers.feature_extraction_utils import PreTrainedFeatureExtractor
from transformers.image_processing_utils import BaseImageProcessor

from src.options.options import Options, Devices
import torch
from typing import Optional, Union, List, Dict, Tuple, Any, Callable
from transformers import pipeline, PreTrainedModel, TFPreTrainedModel, PretrainedConfig, PreTrainedTokenizer, \
    PreTrainedTokenizerFast


class OptionsTextGeneration(Options):
    """
    Options for text-Generation models
    """

    task: str = None
    model: Optional[Union[str, PreTrainedModel, "TFPreTrainedModel"]] = None
    config: Optional[Union[str, PretrainedConfig]] = None
    tokenizer: Optional[Union[str, PreTrainedTokenizer, "PreTrainedTokenizerFast"]] = None
    feature_extractor: Optional[Union[str, PreTrainedFeatureExtractor]] = None
    image_processor: Optional[Union[str, BaseImageProcessor]] = None
    framework: Optional[str] = None
    revision: Optional[str] = None
    use_fast: bool = True
    token: Optional[Union[str, bool]] = None
    device: Optional[Union[int, str, "torch.device"]] = None
    device_map = None
    torch_dtype = None
    trust_remote_code: Optional[bool] = None
    model_kwargs: Dict[str, Any] = None
    pipeline_class: Optional[Any] = None

    def __init__(self, device: Devices,
                 task: str,
                 model: Optional[Union[str, PreTrainedModel, "TFPreTrainedModel"]],
                 config: Optional[Union[str, PretrainedConfig]],
                 tokenizer: Optional[Union[str, PreTrainedTokenizer, "PreTrainedTokenizerFast"]],
                 feature_extractor: Optional[Union[str, PreTrainedFeatureExtractor]],
                 image_processor: Optional[Union[str, BaseImageProcessor]],
                 framework: Optional[str],
                 revision: Optional[str] = None,
                 use_fast: bool = True,
                 token: Optional[Union[str, bool]] = None,
                 device_map=None,
                 torch_dtype=None,
                 trust_remote_code: Optional[bool] = None,
                 model_kwargs: Dict[str, Any] = None,
                 pipeline_class: Optional[Any] = None,
                ):
        """
        Initializes the OptionsTextGeneration
        :param device: The device to use generate prompt
        :param prompt: The prompt to give to the model
        :param max_length: The max length of the generated response
        :param temperature: parameter used during the
         sampling process to control the randomness of generated text
         High temp : High randomness, Low temp : Low randomness
        """
        super().__init__(device)
        # Initialize additional attributes
        self.task = task
        if model:
            self.model = model
        if config:
            self.config = config
        if tokenizer:
            self.tokenizer = tokenizer
        if feature_extractor:
            self.feature_extractor = feature_extractor
        if image_processor:
            self.image_processor = image_processor
        if framework:
            self.framework = framework
        if revision:
            self.revision = revision
        self.use_fast = use_fast
        if token:
            self.token = token
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        if trust_remote_code:
            self.trust_remote_code = trust_remote_code
        self.model_kwargs = model_kwargs
        if pipeline_class  :
            self.pipeline_class = pipeline_class
