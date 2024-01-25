from diffusers import DiffusionPipeline, StableDiffusionXLPipeline
import torch
from src.models.models import Models
from src.options.options_text_to_image import OptionsTextToImage


class ModelsTextToImage(Models):
    pipeline: StableDiffusionXLPipeline
    model_name: str
    loaded: bool

    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.loaded = False
        self.create_pipeline()

    def create_pipeline(self):
        if self.loaded:
            return

        self.pipeline = DiffusionPipeline.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            use_safetensors=True,
            add_watermarker=False,
            variant="fp16",
        )

    def load_model(self, option: OptionsTextToImage):
        if self.loaded:
            return
        self.pipeline.to(option.device)
        self.loaded = True

    def unload_model(self):
        if not self.loaded:
            return
        self.pipeline.to(device="meta")
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        self.loaded = False

    def generate_prompt(self, option: OptionsTextToImage):
        return self.pipeline(prompt=option.prompt, width=option.image_width, height=option.image_height).images[0]
