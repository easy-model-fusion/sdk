from sdk.options import Devices
from transformers import AutoTokenizer, AutoModelForCausalLM

from sdk.models import ModelTransformers


class DemoDatabricks:

    def __init__(self):
        model_path = "databricks/dbrx-instruct"
        tokenizer_path = "databricks/dbrx-instruct"

        model_transformers = ModelTransformers(
            model_name="model",
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            task="text-generation",
            model_class=AutoModelForCausalLM,
            tokenizer_class=AutoTokenizer,
            device=Devices.GPU
        )

        model_transformers.load_model()

        result = model_transformers.generate_prompt(
            prompt="What does it take to build a great LLM?",
            max_length=300,
            truncation=True
        )

        print(result)
