from sdk.options import Devices
from transformers import AutoTokenizer, AutoModelForCausalLM

from sdk.models import ModelTransformers


class DemoGoogleGemma:

    def __init__(self):
        model_path = "google/gemma-2b-it"
        tokenizer_path = "google/gemma-2b-it"

        model_transformers = ModelTransformers(
            model_name="model",
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            task="text-generation",
            model_class=AutoModelForCausalLM,
            tokenizer_class=AutoTokenizer,
            device=Devices.GPU
        )

        model_transformers.create_pipeline()
        model_transformers.load_model()

        result = model_transformers.generate_prompt(
            prompt="Write me a poem about Machine Learning.",
            max_length=300,
            truncation=True
        )

        print(result)
