from sdk.models import ModelsManagement, ModelsTextConversation
from sdk.options import Devices, OptionsTextConversation
from sdk.options.tokenizer_options import TokenizerOptions


class DemoTextConv:

    def __init__(self):
        model_name = "Salesforce/codegen-350M-nl"
        model_path = "Salesforce/codegen-350M-nl"

        options = OptionsTextConversation(
            prompt="Hello, whats 3 + 3 ?",
            device=Devices.GPU,
            model=model_name,
            batch_size=5,
            minimum_tokens=50
        )
        tokenizer_options = TokenizerOptions(
            device='cuda',
            padding_side='left'
        )

        model_management = ModelsManagement()
        model = ModelsTextConversation(model_name, model_path)
        model.tokenizer_options = tokenizer_options
        model_management.add_model(new_model=model, model_options=options)
        model_management.load_model(model_name)

        print(model_management.generate_prompt(options.prompt))
        print(model_management.generate_prompt(prompt="What did I say before ?"))

        options2 = OptionsTextConversation(
            prompt="Hello, whats 6 + 6 ?",
            device=Devices.GPU,
            model=model_name,
            batch_size=1,
            minimum_tokens=50,
            create_new_conv= True
        )
        #Changing to new conversation
        model_management.set_model_options(model_name=model_name
                                           , options=options2)
        print(model_management.generate_prompt(options2.prompt))
        print(model_management.generate_prompt(prompt="Wowowowow ?"))

        #going back to initial conversation
        options2.chat_ID_to_use_id = 0
        print(model_management.generate_prompt("How are you ? "))
        print(model_management.generate_prompt(prompt="Hihi ?"))

        options2.create_new_tokenizer = True
        tokenizer_options = TokenizerOptions(
                    device='cuda',
                    padding_side='right'
                )
        model.tokenizer_options = tokenizer_options
        print(model_management.generate_prompt("Where is Japan"))

