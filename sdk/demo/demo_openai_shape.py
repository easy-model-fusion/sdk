import torch
from sdk.models import ModelDiffusers, ModelsManagement
from sdk.options import Devices
from diffusers.utils import export_to_gif


class DemoOpenaiShape:

    def __init__(self):
        model_openai_shape_name = "openai/shap-e"
        model_openai_shape_path = "openai/shap-e"
        model_management = ModelsManagement()
        model_openai_shape = ModelDiffusers(model_openai_shape_name,
                                              model_openai_shape_path,
                                              Devices.GPU,
                                              torch_dtype=torch.float16,
                                              use_safetensors=False,
                                              add_watermarker=False,
                                              variant="fp16")

        model_management.add_model(new_model=model_openai_shape)
        model_management.load_model(model_openai_shape_name)

        gif = model_management.generate_prompt(
            prompt="a banana",
            guidance_scale= 15.0,
            num_inference_steps=64
        )
        export_to_gif(gif, "3d.gif")