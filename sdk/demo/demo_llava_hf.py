from sdk.options import Devices
from transformers import LlavaForConditionalGeneration, AutoTokenizer

from sdk.models import ModelTransformers
from PIL import Image
import requests


class DemoLlavaHf:

    def __init__(self):

        model_path = "llava-hf/llava-1.5-7b-hf"
        tokenizer_path = "llava-hf/llava-1.5-7b-hf"

        model_transformers = ModelTransformers(
            model_name="model",
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            task="image-to-text",
            model_class=LlavaForConditionalGeneration,
            tokenizer_class=AutoTokenizer,
            device=Devices.GPU
        )

        model_transformers.load_model()

        image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(image_file, stream=True).raw)

        result = model_transformers.generate_prompt(
            prompt="USER: <image>\nWhat are these?\nASSISTANT:",
            max_length=300,
            raw_image=image,
            truncation=True
        )

        print(result)
