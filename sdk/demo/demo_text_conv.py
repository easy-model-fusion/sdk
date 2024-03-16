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
        # Define the model name and path
        model_name = "Salesforce/codegen-350M-nl"
        model_path = "Salesforce/codegen-350M-nl"

        # Define options for text conversation
        options = OptionsTextConversation(
            prompt="Hello, what's 3 + 3 ?",
            device=Devices.GPU,
            model_name=model_name,
            trust_remote_code=True
        )

        # Define tokenizer options
        tokenizer_options = OptionsTokenizer(
            device=Devices.GPU,
            padding_side='left',
            return_tensors="pt"
        )

        tokenizer = Tokenizer("Salesforce/codegen-350M-nl",
                              "Salesforce/codegen-350M-nl",
                              tokenizer_options)

        # Initialize the model management
        model_management = ModelsManagement()

        # Create and load the text conversation model
        model = ModelsTextConversation(model_name, model_path, options)
        model.tokenizer = tokenizer
        model_management.add_model(new_model=model, model_options=options)
        model_management.load_model(model_name)

        # Print the response to the initial prompt
        print(model_management.generate_prompt(options.prompt))

        # Generate a response to a custom prompt
        print(model_management.generate_prompt(
            prompt="What did I say before ?"))

        # Create a new conversation with a new prompt
        options = OptionsTextConversation(
            prompt="Hello, what's 6 + 6 ?",
            device=options.device,
            model_name=model_name,
            batch_size=1,
            chat_id_to_use=1,
            minimum_tokens=50,
            create_new_conv=True
        )

        # Switch to the new conversation
        model_management.set_model_options(model_name=model_name,
                                           options=options)
        print(model_management.generate_prompt(options.prompt))

        print(model_management.generate_prompt("Where is Japan"))

        # Switch back to the initial conversation
        options.chat_id_to_use = 0
        # Create a new tokenizer and use it
        options.tokenizer_id_to_use = 1
        model_management.set_model_options(model_name=model_name,
                                           options=options)
        tokenizer_options = OptionsTokenizer(
            device='cuda',
            padding_side='right',
            return_tensors='pt'
        )
        model.tokenizer_options = tokenizer_options
        print(model_management.generate_prompt("Bye "))
