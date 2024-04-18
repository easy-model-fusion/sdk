from sdk.options import Devices
from transformers import (LlavaForConditionalGeneration, AutoProcessor,
                          AutoTokenizer)

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

        processor = AutoProcessor.from_pretrained(model_path)
        model_transformers.create_pipeline(image_processor=processor)
        model_transformers.load_model()

        image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get
                           (image_url, stream=True).raw).convert('RGB')

        result = model_transformers.generate_prompt(
            prompt=image
        )

        print(result)
