import torch
from sdk.models import ModelsManagement, ModelsTextConversation
from sdk.options import Devices, OptionsTextConversation
from sdk.options.tokenizer_options import TokenizerOptions


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
            model=model_name,
            batch_size=5,
            minimum_tokens=50
        )

        # Define tokenizer options
        tokenizer_options = TokenizerOptions(
            device='cuda',
            padding_side='left'
        )

        # Initialize the model management
        model_management = ModelsManagement()

        # Create and load the text conversation model
        model = ModelsTextConversation(model_name, model_path)
        model.tokenizer_options = tokenizer_options
        model_management.add_model(new_model=model, model_options=options)
        model_management.load_model(model_name)

        # Print the response to the initial prompt
        print(model_management.generate_prompt(options.prompt))

        # Generate a response to a custom prompt
        print(model_management.generate_prompt(
            prompt="What did I say before ?"))

        # Create a new conversation with a new prompt
        options2 = OptionsTextConversation(
            prompt="Hello, what's 6 + 6 ?",
            device=Devices.GPU,
            model=model_name,
            batch_size=1,
            minimum_tokens=50,
            create_new_conv=True
        )

        # Switch to the new conversation
        model_management.set_model_options(model_name=model_name, options=options2)
        print(model_management.generate_prompt(options2.prompt))
        print(model_management.generate_prompt(
            prompt="Wowowowow ?")
        )

        # Switch back to the initial conversation
        options2.chat_ID_to_use_id = 0
        print(model_management.generate_prompt("How are you ? "))
        print(model_management.generate_prompt(prompt="Hihi ?"))

        # Create a new tokenizer and use it
        options2.create_new_tokenizer = True
        tokenizer_options = TokenizerOptions(
            device='cuda',
            padding_side='right'
        )
        model.tokenizer_options = tokenizer_options
        print(model_management.generate_prompt("Where is Japan"))

