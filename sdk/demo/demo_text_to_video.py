import torch
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video

from sdk.models import ModelDiffusers, ModelsManagement
from sdk.options import Devices


class DemoTextToVideo:
    """
    This class demonstrates text generation using a TextToVideo model.

    """
    def __init__(self):
        """
        Initializes the DemoTextGen class with predefined options and models
        and returns result of prompt  .
        """
        model_name = "damo-vilab/text-to-video-ms-1.7b"
        model_path = "damo-vilab/text-to-video-ms-1.7b"

        model_options = {
            'torch_dtype': torch.float16,
            'use_safetensors': True,
            'add_watermarker': False,
            'variant': "fp16"
        }

        model_management = ModelsManagement()
        model = ModelDiffusers(
            model_name=model_name,
            model_path=model_path,
            model_class=DiffusionPipeline,
            device=Devices.GPU,
            **model_options)

        model_management.add_model(new_model=model)
        model_management.load_model(model_name)

        prompt = ("Astronaut in a jungle, cold color palette,"
                  " muted colors, detailed, 8k")

        video_frames = model_management.generate_prompt(
            prompt=prompt,
            model_name=model_name,
            num_inference_steps=25
        ).frames

        video_path = export_to_video(video_frames)

        print(video_path)
