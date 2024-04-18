import torch
from sdk.models import ModelDiffusers, ModelsManagement
from sdk.options import Devices
from diffusers import StableDiffusionPipeline


class DemoRunwayml:

    def __init__(self):
        model_runwayml_name = "runwayml/stable-diffusion-v1-5"
        model_runwayml_path = "runwayml/stable-diffusion-v1-5"
        model_management = ModelsManagement()
        model_runwayml = ModelDiffusers(model_name=model_runwayml_name,
                                        model_path=model_runwayml_path,
                                        device=Devices.GPU,
                                        torch_dtype=torch.float16,
                                        use_safetensors=True,
                                        add_watermarker=False,
                                        model_class=StableDiffusionPipeline)

        model_management.add_model(new_model=model_runwayml)
        model_management.load_model(model_runwayml_name)

        image = model_management.generate_prompt(
            prompt="a photo of a garage with luxury car",
            model_name=model_runwayml_name,
            image_width=512,
            image_height=512
        ).images[0]
        image.show()
