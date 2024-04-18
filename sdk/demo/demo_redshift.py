import torch
from sdk.models import ModelDiffusers, ModelsManagement
from sdk.options import Devices


class DemoRedshift:

    def __init__(self):
        model_redshift_name = "nitrosocke/redshift-diffusion"
        model_redshift_path = "nitrosocke/redshift-diffusion"
        model_management = ModelsManagement()
        model_redshift = ModelDiffusers(model_redshift_name,
                                             model_redshift_path,
                                             Devices.GPU,
                                             torch_dtype=torch.float16,
                                             use_safetensors=False,
                                             add_watermarker=False)

        model_management.add_model(new_model=model_redshift)
        model_management.load_model(model_redshift_name)

        image = model_management.generate_prompt(
            prompt="redshift style magical princess with golden hair",
            image_width=512,
            image_height=512
        )
        image.show()
