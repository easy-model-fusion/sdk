import torch
from diffusers import DiffusionPipeline

from sdk.models import ModelDiffusers, ModelsManagement
from sdk.options import Devices


class DemoTextToImg:

    def __init__(self):
        model_stabilityai_name = "stabilityai/sdxl-turbo"
        model_stabilityai_path = "stabilityai/sdxl-turbo"

        model_options = {
            'torch_dtype': torch.float16,
            'use_safetensors': True,
            'add_watermarker': False,
            'variant': "fp16"
        }

        model_management = ModelsManagement()
        model_stabilityai = ModelDiffusers(
            model_name=model_stabilityai_name,
            model_path=model_stabilityai_path,
            model_class=DiffusionPipeline,
            device=Devices.GPU,
            **model_options)

        model_management.add_model(new_model=model_stabilityai)
        model_management.load_model(model_stabilityai_name)

        image = model_management.generate_prompt(
            prompt="Astronaut in a jungle, cold color palette, "
                   "muted colors, detailed, 8k",
            image_width=512,
            image_height=512
        ).images[0]
        image.show()
