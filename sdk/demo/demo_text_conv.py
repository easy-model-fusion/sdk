from sdk import ModelsManagement
from sdk.models import ModelsTextConversation
from sdk.options import Devices
from transformers import AutoTokenizer, AutoModelForCausalLM


class DemoTextConv:
    """
    This class demonstrates a text conversation using a chatbot model.

    Attributes:
        model_path (str): The path to the model.
        tokenizer_path (str): The path to the tokenizer.
        model_management (ModelsManagement): An instance of
            the ModelsManagement class.
        model (ModelsTextConversation): An instance of the
            ModelsTextConversation class.
    """

    def __init__(self):
        """
        __init__ Initializes the DemoTextConv class with predefined options and models
        and returns result of prompt .
        """
        model_path = "microsoft/phi-2"
        tokenizer_path = "microsoft/phi-2"

        model_management = ModelsManagement()

        model = ModelsTextConversation(model_name="model",
                                       model_path=model_path,
                                       tokenizer_path=tokenizer_path,
                                       model_class=AutoModelForCausalLM,
                                       tokenizer_class=AutoTokenizer,
                                       device=Devices.GPU)

        model.create_pipeline()
        model_management.add_model(new_model=model)
        model_management.load_model(model.model_name)
        model.schematic["role"] = "user"
        model.create_new_conversation()

        result = model_management.generate_prompt(
            "I'm looking for a movie - what's your favourite one?",
            model_name=model.model_name)

        print(result.messages[-1]["content"])
