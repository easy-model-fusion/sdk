import torch
from typing import Union, Any

from sdk.options import Devices
from sdk.models import Model


class ModelDiffusers(Model):
    """
    This class implements methods to generate with the diffusers models.

    Attributes:
        model_class (Any): The class of the model used for interaction.
        pipeline (Any): The pipeline object for the model.
    """

    model_class: Any
    pipeline: Any

    def __init__(self, model_name: str,
                 model_path: str,
                 model_class: Any,
                 device: Union[str, Devices],
                 **kwargs):
        """
        Initializes the ModelsTextToImage class.

        Args:
            model_name (str): The name of the model.
            model_path (str): The path of the model.
            model_class (Any): The model class use to interact with the model.
            device (Union[str, Devices]): Which device the model must be on.
            **kwargs: Parameters for the model.
        """
        super().__init__(model_name, model_path, device)
        self.model_class = model_class
        self.create_pipeline(**kwargs)

    def create_pipeline(self, **kwargs) -> None:
        """
        Creates the pipeline to load on the device.

        Args:
            **kwargs: Parameters for the model.
        """
        if self.loaded:
            return

        # Single file loads a single ckpt/safetensors file
        if self.single_file:
            self.pipeline = self.model_class.from_single_file(
                self.model_path,
                **kwargs
            )
            return

        self.pipeline = self.model_class.from_pretrained(
            self.model_path,
            **kwargs
        )

    def load_model(self) -> bool:
        """
        Load this model on the given device.

        Returns:
             bool: True if the model is successfully loaded.
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
        Unloads the model.

        Returns:
             bool: True if the model is successfully unloaded.
        """
        if not self.loaded:
            return False
        self.pipeline.to(device=Devices.RESET.value)
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        self.loaded = False
        return True

    def generate_prompt(self, prompt: Any,
                        **kwargs):
        """
        Generates the prompt with the given option.

        Args:
            prompt (Any): The prompt to generate.

        Returns:
             An object image resulting from the model.
        """
        return self.pipeline(
            prompt=prompt,
            **kwargs
        )

