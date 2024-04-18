import requests
import torch
from PIL import Image
from transformers import (BlipProcessor, BlipForConditionalGeneration,
                          AutoTokenizer)
from sdk.options import Devices
from sdk.models import ModelTransformers


class DemoSalesforce:

    def __init__(self):
        model_path = "Salesforce/blip-image-captioning-large"
        tokenizer_path = "Salesforce/blip-image-captioning-large"

        model_transformers = ModelTransformers(
            model_name="blip-image-captioning",
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            task="image-to-text",
            model_class=BlipForConditionalGeneration,
            tokenizer_class=AutoTokenizer,
            device=Devices.GPU
        )

        processor = BlipProcessor.from_pretrained(model_path)
        model_transformers.create_pipeline(image_processor=processor)
        model_transformers.load_model()

        image_url = ('https://storage.googleapis.com/'
                     'sfr-vision-language-research/BLIP/demo.jpg')
        raw_image = Image.open(requests.get
                               (image_url, stream=True).raw).convert('RGB')

        result = model_transformers.generate_prompt(
            prompt=raw_image
        )
        print(result)
