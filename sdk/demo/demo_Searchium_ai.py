from sdk.options import Devices
from transformers import CLIPTokenizer, CLIPTextModelWithProjection

from sdk.models import ModelTransformers


class DemoSearchiumAi:

    def __init__(self):
        model_path = "Searchium-ai/clip4clip-webvid150k"
        tokenizer_path = "Searchium-ai/clip4clip-webvid150k"

        model_transformers = ModelTransformers(
            model_name="model",
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            task="text-to-video",
            model_class=CLIPTextModelWithProjection,
            tokenizer_class=CLIPTokenizer,
            device=Devices.GPU
        )

        model_transformers.create_pipeline()
        model_transformers.load_model()

        result = model_transformers.generate_prompt(
            prompt="a basketball player performing a slam dunk"
        )

        final_output = result[0] / result[0].norm(dim=-1, keepdim=True)
        final_output = final_output.cpu().detach().numpy()
        print("final output: ", final_output)
