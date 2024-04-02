from sdk.models import ModelsTextConversation
from sdk.options import Devices
from transformers import AutoTokenizer, AutoModelForCausalLM


class DemoTextConv:
    """
    This class demonstrates a text conversation using a chatbot model.
    """

    def __init__(self):
        """
        Initializes the DemoTextConv class with predefined options and models.
        """
        model_path = "microsoft/phi-2"
        tokenizer_path = "microsoft/phi-2"

        model = ModelsTextConversation(model_name="model",
                                       model_path=model_path,
                                       tokenizer_path=tokenizer_path,
                                       model_class=AutoModelForCausalLM,
                                       tokenizer_class=AutoTokenizer,
                                       device=Devices.GPU)
        model.create_pipeline()
        model.load_model()
        model.schematic["role"] = "user"
        model.create_new_conversation()

        result = model.generate_prompt(
            "I'm looking for a movie - what's your favourite one?")

        print(result.messages[-1]["content"])
