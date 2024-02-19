import torch
from src.models.models_management import ModelsManagement
from src.models.models_text_conversation import ModelsTextConversation
from src.options.options import Devices
from src.options.options_text_conversation import OptionsTextConversation


class DemoMainConv():


    def __init__(self):
        model_stabilityai_name = "facebook/blenderbot-400M-distill"

        options = OptionsTextConversation(
            prompt="Hello",
            device=Devices.CPU,
            model=model_stabilityai_name,
            batch_size=3,
            minimum_tokens=50
        )

        model_management = ModelsManagement()
        model_stabilityai = ModelsTextConversation(model_stabilityai_name)

        model_management.add_model(new_model=model_stabilityai, model_options=options)
        model_management.load_model(model_stabilityai_name)

        model = ModelsTextConversation(model_stabilityai_name)
        model.generate_prompt(options.prompt, options)
        self.demo(model)

    def demo(self, model: ModelsTextConversation):
        print("User : Hello")
        new_user_input_ids = model.tokenizer.encode("Hello ! " + model.tokenizer.eos_token, return_tensors='pt')
        # append the new user input tokens to the chat history
        bot_input_ids = torch.cat([model.chat_history_ids, new_user_input_ids],
                                  dim=-1) if model.conversation_step > 0 else new_user_input_ids
        # generated a response while limiting the total chat history to 1000 tokens,
        chat_history_ids = model.pipeline.generate(bot_input_ids, max_length=1000,
                                                   pad_token_id=model.tokenizer.eos_token_id)
        print("Chatbot: {}".format(
            model.tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))
        print("User : How are you ?")
        new_user_input_ids = model.tokenizer.encode("How are you ? " + model.tokenizer.eos_token, return_tensors='pt')
        # append the new user input tokens to the chat history
        bot_input_ids = torch.cat([model.chat_history_ids, new_user_input_ids],
                                  dim=-1) if model.conversation_step > 0 else new_user_input_ids
        # generated a response while limiting the total chat history to 1000 tokens,
        chat_history_ids = model.pipeline.generate(bot_input_ids, max_length=1000,
                                                   pad_token_id=model.tokenizer.eos_token_id)
        print("Chatbot: {}".format(
            model.tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))
