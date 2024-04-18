import torch
from sdk.models import ModelDiffusers, ModelsManagement
from sdk.options import Devices


class DemoDreamLikeArt:

    def __init__(self):
        model_dreamlike_art_name = "dreamlike-art/dreamlike-diffusion-1.0"
        model_dreamlike_art_path = "dreamlike-art/dreamlike-diffusion-1.0"
        model_management = ModelsManagement()
        model_dreamlike_art = ModelDiffusers(model_dreamlike_art_name,
                                             model_dreamlike_art_path,
                                             Devices.GPU,
                                             torch_dtype=torch.float16,
                                             use_safetensors=True,
                                             add_watermarker=False)

        model_management.add_model(new_model=model_dreamlike_art)
        model_management.load_model(model_dreamlike_art_name)

        image = model_management.generate_prompt(
            prompt="dreamlikeart, a grungy woman with rainbow hair",
            image_width=512,
            image_height=512
        )
        image.show()
