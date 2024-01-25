from diffusers import DiffusionPipeline, StableDiffusionXLPipeline
import torch
from src.models.models import Models
from src.options.options_text_to_image import OptionsTextToImage, Devices


class ModelsTextToImage(Models):
    """
    This class implements methods to generate images with a text prompt
    """
    pipeline: StableDiffusionXLPipeline
    model_name: str
    loaded: bool

    def __init__(self, model_name: str):
        """
        Initializes the ModelsTextToImage class
        :param model_name: The name of the model
        """
        super().__init__(model_name)
        self.loaded = False
        self.create_pipeline()

    def create_pipeline(self):
        """
        Creates the pipeline to load on the device
        """
        if self.loaded:
            return

        self.pipeline = DiffusionPipeline.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            use_safetensors=True,
            add_watermarker=False,
            variant="fp16",
        )

    def load_model(self, option: OptionsTextToImage) -> bool:
        """
        Load this model on the given device
        :param option: The options with the device
        :return: True if the model is successfully loaded
        """
        if self.loaded:
            return True
        if option.device == Devices.RESET:
            return False
        self.pipeline.to(option.device)
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

    def generate_prompt(self, options: OptionsTextToImage):
        """
        Generates the prompt with the given option
        :param options: The options of text to image model
        :return: An object image resulting from the model
        """
        return self.pipeline(
            prompt=options.prompt,
            width=options.image_width,
            height=options.image_height).images[0]
