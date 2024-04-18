import torch
from sdk.models import ModelDiffusers, ModelsManagement
from sdk.options import Devices
from diffusers import StableDiffusionPipeline


class DemoRedshift:

    def __init__(self):
        model_redshift_name = "nitrosocke/redshift-diffusion"
        model_redshift_path = "nitrosocke/redshift-diffusion"
        model_management = ModelsManagement()
        model_redshift = ModelDiffusers(model_name=model_redshift_name,
                                        model_path=model_redshift_path,
                                        device=Devices.GPU,
                                        torch_dtype=torch.float16,
                                        use_safetensors=False,
                                        add_watermarker=False,
                                        model_class=StableDiffusionPipeline)

        model_management.add_model(new_model=model_redshift)
        model_management.load_model(model_redshift_name)

        image = model_management.generate_prompt(
            prompt="redshift style magical princess with golden hair",
            model_name=model_redshift_name,
            image_width=512,
            image_height=512
        ).images[0]
        image.show()
