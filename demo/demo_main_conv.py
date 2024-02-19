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

        model.create_pipeline()
        model.generate_prompt(options)
        def add_input(self):
            new_user_input_ids = self.tokenizer.encode(input(">> User:") + self.tokenizer.eos_token, return_tensors='pt')
            # append the new user input tokens to the chat history
            bot_input_ids = torch.cat([self.chat_history_ids, new_user_input_ids],
                                      dim=-1) if self.conversation_step > 0 else new_user_input_ids
            # generated a response while limiting the total chat history to 1000 tokens,
            chat_history_ids = self.pipeline.generate(bot_input_ids, max_length=1000,
                                                      pad_token_id=self.tokenizer.eos_token_id)
            print("Chatbot: {}".format(
                self.tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))
            self.conversation_step += 1