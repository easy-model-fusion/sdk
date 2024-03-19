from sdk.models import ModelsManagement, ModelsTextConversation
from sdk.options import Devices, OptionsTextConversation
from sdk.options.options_tokenizer import OptionsTokenizer
from sdk.tokenizers.tokenizer import Tokenizer


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

        tokenizer = Tokenizer(tokenizer_name="token",
                              tokenizer_path=tokenizer_path,
                              device=Devices.GPU)

        model = ModelsTextConversation(model_name="model",
                                       model_path=model_path,
                                       tokenizer=tokenizer,
                                       device=Devices.GPU)

        model.load_model()
        model.create_new_conversation()

        result = model.generate_prompt(
            "I'm looking for a movie - what's your favourite one?")

        print(result.messages[-1]["content"])

