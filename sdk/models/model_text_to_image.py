import torch
from diffusers import DiffusionPipeline, StableDiffusionXLPipeline
from sdk.options import Devices
from typing import Union
from sdk.models import Model


class ModelTextToImage(Model):
    """
    This class implements methods to generate images with a text prompt
    """
    pipeline: StableDiffusionXLPipeline
    loaded: bool
    device: Union[str, Devices]

    def __init__(self, model_name: str, model_path: str,
                 device: Union[str, Devices], **kwargs):
        """
        Initializes the ModelsTextToImage class
        :param model_name: The name of the model
        :param model_path: The path of the model
        """
        super().__init__(model_name, model_path)
        self.device = device
        self.loaded = False
        self.create_pipeline(**kwargs)

    def create_pipeline(self, **kwargs):
        """
        Creates the pipeline to load on the device
        """
        if self.loaded:
            return

        self.pipeline = DiffusionPipeline.from_pretrained(
            self.model_path,
            **kwargs
        )

    def load_model(self) -> bool:
        """
        Load this model on the given device
        :return: True if the model is successfully loaded
        """
        if self.loaded:
            return True
        if self.device == Devices.RESET:
            return False
        self.pipeline.to(device=(
            self.device if isinstance(self.device, str) else (
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
        self.pipeline.to(device=Devices.RESET)
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        self.loaded = False
        return True

    def generate_prompt(self, prompt: str,
                        **kwargs):
        """
        Generates the prompt with the given option
        :param prompt: The prompt to generate
        :return: An object image resulting from the model
        """
        return self.pipeline(
            prompt=prompt,
            **kwargs
        ).images[0]
