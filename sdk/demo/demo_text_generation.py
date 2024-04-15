from sdk import ModelsManagement
from sdk.options import Devices
from transformers import AutoTokenizer, AutoModelForCausalLM

from sdk.models import ModelTransformers


class DemoTextGen:
    """
    This class demonstrates a text conversation using a chatbot model.
    """

    def __init__(self):
        """
        Initializes the DemoTextConv class with predefined options and models.
        """
        model_path = "microsoft/phi-2"
        tokenizer_path = "microsoft/phi-2"

        model_management = ModelsManagement()

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
        model_management.add_model(new_model=model_transformers)
        model_management.load_model(model_transformers.model_name)

        result = model_management.generate_prompt(
            prompt="I'm looking for a movie - what's your favourite one?",
            max_length=300,
            truncation=True,
            model_name=model_transformers.model_name
        )

        print(result)
