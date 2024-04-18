import torch
from sdk.models import ModelDiffusers, ModelsManagement
from sdk.options import Devices
from diffusers import AudioLDM2Pipeline
import scipy


class DemoCvssp:

    def __init__(self):
        model_cvssp_name = "cvssp/audioldm2"
        model_cvssp_path = "cvssp/audioldm2"
        model_management = ModelsManagement()
        model_cvssp = ModelDiffusers(model_name=model_cvssp_name,
                                     model_path=model_cvssp_path,
                                     device=Devices.GPU,
                                     torch_dtype=torch.float16,
                                     use_safetensors=True,
                                     add_watermarker=False,
                                     model_class=AudioLDM2Pipeline)

        model_management.add_model(new_model=model_cvssp)
        model_management.load_model(model_cvssp_name)

        audio = model_management.generate_prompt(
            prompt="The sound of a hammer hitting a wooden surface",
            model_name=model_cvssp_name,
            negative_prompt="Low quality.",
            num_inference_steps=200,
            audio_length_in_s=10.0,
            num_waveforms_per_prompt=3,
        ).audios[0]
        scipy.io.wavfile.write("audio.wav", rate=16000, data=audio)
