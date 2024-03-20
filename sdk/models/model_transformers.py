import torch
from typing import Optional, Any, Union
from transformers import (
    pipeline,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from sdk.tokenizers.tokenizer import Tokenizer
from sdk.models import Model
from sdk.options import Devices


class ModelTransformers(Model):
    """
    A class to use if the model is a transformers one
    """
    task: str
    # ToDo: look for a best type
    model_class: Any
    tokenizer_class: Any
    device: Union[str, Devices]

    tokenizer_path: str

    model_pipeline_args: dict[str, Any] = None
    tokenizer_pipeline_args: dict[str, Any] = None

    transformers_pipeline: pipeline
    model_pipeline: PreTrainedModel
    tokenizer_pipeline: PreTrainedTokenizer

    loaded: bool

    def __init__(self, model_name: str, model_path: str,
                 tokenizer_path,
                 task: str,
                 model_class: Any,
                 tokenizer_class: Any,
                 device: Union[str, Devices]
                 ):
        """
        Initializes the ModelsTextToImage class
        :param model_name: The name of the model
        :param model_path: The path of the model
        :param device: Which device the model must be on
        """
        super().__init__(model_name, model_path)
        self.tokenizer_path = tokenizer_path
        self.task = task
        self.device = device
        self.loaded = False
        self.model_class = model_class
        self.tokeniser_class = tokenizer_class
        self.create_pipeline()

    def set_model_pipeline_args(self, **kwargs):
        if kwargs:
            self.model_pipeline_args = kwargs.copy()

    def set_tokenizer_pipeline_args(self, **kwargs):
        if kwargs:
            self.tokenizer_pipeline_args = kwargs.copy()

    def create_pipeline(self, **kwargs) -> None:
        """
        Creates the pipeline and load them on the device
        """
        if self.loaded:
            return

        self.model_pipeline = self.model_class.from_pretrained(
            self.model_path,
            **self.model_pipeline_args
        )

        self.tokenizer_pipeline = self.tokeniser_class.from_pretrained(
            self.tokenizer_path,
            **self.tokenizer_pipeline_args
        )

        self.transformers_pipeline = pipeline(
            task=self.task,
            model=self.model_pipeline,
            tokenizer=self.tokenizer_pipeline,
            device=(
                self.device if isinstance(
                    self.device, str) else (
                    self.device.value)
            ),
            ** kwargs
        )

    def load_model(self) -> bool:
        """
        Load this model on the given device,

        Returns:
            bool: True if the model is successfully loaded.
        """

        if self.loaded:
            return True
        if self.device == Devices.RESET.value:
            return False

        # When the device is turn to meta, we must recreate them to load it
        if (self.transformers_pipeline.device ==
                torch.device(Devices.RESET.value)):
            self.model_pipeline = self.model_class.from_pretrained(
                self.model_path,
                **self.model_pipeline_args
            )

        self.transformers_pipeline.model.to(device=(
            self.device if isinstance(
                self.device, str) else (
                self.device.value)))
        self.loaded = True
        return True

    def unload_model(self) -> bool:
        """
        Unloads the model
        :return: True if the model is successfully unloaded
        """
        if not self.loaded:
            return False

        self.transformers_pipeline.model.to(Devices.RESET.value)
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        self.loaded = False
        return True

    def generate_prompt(
            self, prompt: Any,
            **kwargs) -> Any:
        """
        Generates the prompt with the given option.

        Args:
            prompt (Any): The prompt.

        Returns:
            Any: Generated prompt.
        """
        return self.transformers_pipeline(prompt)
