import torch
from sdk.models import ModelDiffusers, ModelsManagement
from sdk.options import Devices
from diffusers import StableDiffusionPipeline


class DemoDreamLikeArt:

    def __init__(self):
        model_dreamlike_name = "dreamlike-art/dreamlike-diffusion-1.0"
        model_dreamlike_path = "dreamlike-art/dreamlike-diffusion-1.0"
        model_management = ModelsManagement()
        model_dreamlike = ModelDiffusers(model_name=model_dreamlike_name,
                                         model_path=model_dreamlike_path,
                                         device=Devices.GPU,
                                         torch_dtype=torch.float16,
                                         use_safetensors=True,
                                         add_watermarker=False,
                                         model_class=StableDiffusionPipeline)

        model_management.add_model(new_model=model_dreamlike)
        model_management.load_model(model_dreamlike_name)

        image = model_management.generate_prompt(
            prompt="dreamlikeart, a grungy woman with rainbow hair",
            model_name=model_dreamlike_name,
            image_width=512,
            image_height=512
        ).images[0]
        image.show()
