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
            batch_size=3,
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
        model_management.generate_prompt(options.prompt)


