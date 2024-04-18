import torch
from sdk.models import ModelDiffusers, ModelsManagement
from sdk.options import Devices
from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image


class DemoStabilityaiImg:

    def __init__(self):
        model_stab_refiner_name = "stabilityai/stable-diffusion-xl-refiner-1.0"
        model_stab_refiner_path = "stabilityai/stable-diffusion-xl-refiner-1.0"
        model_management = ModelsManagement()

        model_stab_refiner = ModelDiffusers(model_stab_refiner_name,
                                              model_stab_refiner_path,
                                              Devices.GPU,
                                              torch_dtype=torch.float16,
                                              use_safetensors=True,
                                              add_watermarker=False,
                                              variant="fp16")

        model_management.add_model(new_model=model_stab_refiner)
        model_management.load_model(model_stab_refiner_name)

        url = ("https://huggingface.co/datasets/patrickvonplaten/"
               "images/resolve/main/aa_xl/000000009.png")
        init_image = load_image(url).convert("RGB")

        image = model_management.generate_prompt(
            prompt="a photo of an human riding a dog",
            image=init_image,
            image_width=512,
            image_height=512
        )
        image.show()

