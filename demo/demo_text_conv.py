import torch
from src.models.models_management import ModelsManagement
from src.models.models_text_conversation import ModelsTextConversation
from src.options.options import Devices
from src.options.options_text_conversation import OptionsTextConversation


class DemoTextConv:

    def __init__(self):
        model_name = "facebook/blenderbot-400M-distill"

        options = OptionsTextConversation(
            prompt="Hello",
            device=Devices.CPU,
            model=model_name,
            batch_size=3,
            minimum_tokens=50
        )

        model_management = ModelsManagement()
        model_stabilityai = ModelsTextConversation(model_name)

        model_management.add_model(new_model=model_stabilityai, model_options=options)
        model_management.load_model(model_name)

        model_management.model = ModelsTextConversation(model_name)
        model_management.generate_prompt(options.prompt)
        self.demo(model_management.model)

    def demo(self, model_management: ModelsTextConversation):
        print("User : Hello")
        new_user_input_ids = model_management.tokenizer.encode("Hello ! " + model_management.tokenizer.eos_token,
                                                               return_tensors='pt')
        # append the new user input tokens to the chat history
        bot_input_ids = torch.cat([model_management.chat_history_ids, new_user_input_ids],
                                  dim=-1) if model_management.conversation_step > 0 else new_user_input_ids
        # generated a response while limiting the total chat history to 1000 tokens,
        chat_history_ids = model_management.pipeline.generate(bot_input_ids, max_length=1000,
                                                              pad_token_id=model_management.tokenizer.eos_token_id)
        print("Chatbot: {}".format(
            model_management.tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0],
                                              skip_special_tokens=True)))
        print("User : How are you ?")
        new_user_input_ids = model_management.tokenizer.encode("How are you ? " + model_management.tokenizer.eos_token,
                                                               return_tensors='pt')
        # append the new user input tokens to the chat history
        bot_input_ids = torch.cat([model_management.chat_history_ids, new_user_input_ids],
                                  dim=-1) if model_management.conversation_step > 0 else new_user_input_ids
        # generated a response while limiting the total chat history to 1000 tokens,
        chat_history_ids = model_management.pipeline.generate(bot_input_ids, max_length=1000,
                                                              pad_token_id=model_management.tokenizer.eos_token_id)
        print("Chatbot: {}".format(
            model_management.tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0],
                                              skip_special_tokens=True)))
