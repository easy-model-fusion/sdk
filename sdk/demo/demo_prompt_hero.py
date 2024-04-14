import torch
from sdk.models import ModelTextToImage, ModelsManagement
from sdk.options import Devices


class DemoPromptHero:

    def __init__(self):
        model_prompt_hero_name = "prompthero/openjourney"
        model_prompt_hero_path = "prompthero/openjourney"
        model_management = ModelsManagement()
        model_prompt_hero = ModelTextToImage(model_prompt_hero_name,
                                             model_prompt_hero_path,
                                             Devices.GPU,
                                             torch_dtype=torch.float16,
                                             use_safetensors=True,
                                             add_watermarker=False)

        model_management.add_model(new_model=model_prompt_hero)
        model_management.load_model(model_prompt_hero_name)

        image = model_management.generate_prompt(
            prompt="retro serie of different cars with different "
                   "colors and shapes, mdjrny-v4 style",
            image_width=512,
            image_height=512
        )
        image.show()
