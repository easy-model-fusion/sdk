import torch
from typing import Any, Union
from transformers import (
    pipeline,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from sdk.models import Model
from sdk.options import Devices


class ModelTransformers(Model):
    """
    A class to use if the model is a transformers one.

    Attributes:
        tokenizer_path (str): The path of the tokenizer.
        task (str): The parameter represents the model type.
        model_class (Any): The model class used to interact with the model.
        tokenizer_class (Any): The tokenizer class used to interact
            with the model.
        model_pipeline_args (dict): Additional arguments for preparing
            the model pipeline.
        tokenizer_pipeline_args (dict): Additional arguments for preparing
            the tokenizer pipeline.
        transformers_pipeline (pipeline): The pipeline for performing various
            tasks with the model.
        model_pipeline (PreTrainedModel): The pretrained model.
        tokenizer_pipeline (PreTrainedTokenizer): The pretrained tokenizer.
    """

    tokenizer_path: str
    task: str

    model_class: Any
    tokenizer_class: Any

    model_pipeline_args: dict[str, Any] = {}
    tokenizer_pipeline_args: dict[str, Any] = {}

    transformers_pipeline: pipeline
    model_pipeline: PreTrainedModel
    tokenizer_pipeline: PreTrainedTokenizer

    def __init__(self, model_name: str,
                 model_path: str,
                 tokenizer_path: str,
                 task: str,
                 model_class: Any,
                 tokenizer_class: Any,
                 device: Union[str, Devices]
                 ):
        """
        __init__ Initializes the Model Transformers class used to interact
            with the model.

        :param model_name: The name of the model.
        :param model_path: The path of the model.
        :param tokenizer_path: The path of the tokenizer.
        :param task: The parameter represents the model type.
        :param model_class: The model class used to interact
            with the model.
        :param tokenizer_class: The tokenizer class used
            to interact with the model.
        :param device: Which device the model must be on.
        """
        super().__init__(model_name, model_path, device)
        self.tokenizer_path = tokenizer_path
        self.task = task
        self.model_class = model_class
        self.tokeniser_class = tokenizer_class

    def set_model_pipeline_args(self, **kwargs) -> None:

        """
        set_model_pipeline_args Store kwargs to prepare model
            for create_pipeline method.

        :param kwargs: Parameters for model.
        """
        if kwargs:
            self.model_pipeline_args = kwargs.copy()

    def set_tokenizer_pipeline_args(self, **kwargs) -> None:
        """
        set_tokenizer_pipeline_args Stores kwargs to prepare tokenizer
            for create_pipeline method.

        :param kwargs: Parameters for tokenizer.
        """
        if kwargs:
            self.tokenizer_pipeline_args = kwargs.copy()

    def create_pipeline(self, **kwargs) -> None:
        """
        create_pipeline Creates all pipelines and loads them onto the device.

        :param kwargs: Parameters for transformers pipeline.
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
        load_model Loads this model on the given device.

        :return: True if the model is successfully loaded.
        """

        if self.loaded:
            return True
        if self.device == Devices.RESET.value:
            return False

        # When the device is turned to meta, we must recreate them to load it.
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
        unload_model Unloads the model.

        :return: True if the model is successfully unloaded.
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
        generate_prompt Generates the prompt with the given option.

        :param prompt: The prompt.
        :param kwargs: Additional parameters for generating the prompt.

        :return: Generated prompt.
        """
        return self.transformers_pipeline(prompt, **kwargs)
