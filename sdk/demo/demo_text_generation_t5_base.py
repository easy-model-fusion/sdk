from sdk.options import Devices
from transformers import T5Tokenizer, T5ForConditionalGeneration

from sdk.models import ModelTransformers


class DemoT5Base:

    def __init__(self):
        model_path = "t5-base"
        tokenizer_path = "t5-base"

        model_transformers = ModelTransformers(
            model_name="model",
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            task="text-generation",
            model_class=T5ForConditionalGeneration,
            tokenizer_class=T5Tokenizer,
            device=Devices.GPU
        )

        model_transformers.create_pipeline()
        model_transformers.load_model()

        result = model_transformers.generate_prompt(
            prompt="Studies have been shown that owning a dog is good for you",
            max_length=300,
            truncation=True
        )

        print(result)
