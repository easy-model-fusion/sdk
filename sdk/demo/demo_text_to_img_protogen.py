import torch
from sdk.models import ModelDiffusers, ModelsManagement
from sdk.options import Devices
from diffusers import StableDiffusionPipeline


class DemoProtogen:

    def __init__(self):
        model_protogen_name = "darkstorm2150/Protogen_v2.2_Official_Release"
        model_protogen_path = "darkstorm2150/Protogen_v2.2_Official_Release"
        model_management = ModelsManagement()
        model_protogen = ModelDiffusers(model_name=model_protogen_name,
                                        model_path=model_protogen_path,
                                        device=Devices.GPU,
                                        torch_dtype=torch.float16,
                                        use_safetensors=False,
                                        add_watermarker=False,
                                        model_class=StableDiffusionPipeline)

        model_management.add_model(new_model=model_protogen)
        model_management.load_model(model_protogen_name)

        image = model_management.generate_prompt(
            prompt="modelshoot style, (extremely detailed CG unity 8k"
                   "wallpaper), full shot body photo of the most beautiful"
                   "artwork in the world, english medieval witch, black silk"
                   "vale, pale skin, black silk robe, black cat, necromancy"
                   "magic, medieval era, photorealistic painting by Ed Blinkey"
                   ", Atey Ghailan, Studio Ghibli, by Jeremy Mann, Greg"
                   " Manchess, Antonio Moro, trending on ArtStation, "
                   "trending on CGSociety, Intricate, High Detail, Sharp focus"
                   ", dramatic, photorealistic painting art by midjourney and"
                   "greg rutkowski",
            model_name=model_protogen_name,
            image_width=512,
            image_height=512
        ).images[0]
        image.show()
