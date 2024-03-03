import torch
from sdk.models import ModelsManagement, ModelsTextConversation
from sdk.options import Devices, OptionsTextConversation
from sdk.options.fast_pretrained_tokenizer_options import FastPreTrainedTokenizerOptions
from sdk.tokenizers.pre_trained_fast import PreTrainedFast


class DemoTextConv:

    def __init__(self):
        model_name = "facebook/blenderbot-400M-distill"
        model_path = "facebook/blenderbot-400M-distill"

        options = OptionsTextConversation(
            prompt="Hello",
            device=Devices.GPU,
            model=model_name,
            batch_size=3,
            minimum_tokens=50
        )
        tokenizer_options = FastPreTrainedTokenizerOptions(
            device=Devices.GPU
        )

        model_management = ModelsManagement()
        model = ModelsTextConversation(model_name, model_path)
        model.tokenizer_options = tokenizer_options
        model.tokenizer = PreTrainedFast(
            model_name="facebook/blenderbot-400M-distill",
            model_path="facebook/blenderbot-400M-distill",
            tokenizer_options=tokenizer_options
        )
        model_management.add_model(new_model=model, model_options=options)
        model_management.load_model(model_name)

        model_management.generate_prompt(options.prompt)
        if model_management.loaded_model:
            self.demo(model_management.loaded_model)

    def demo(self, model: ModelsTextConversation):
        print("Model name : ", model.model_name)
        print("User : Hello")

        new_user_input_ids = model.tokenizer.encode(
            "Hello ! "
            + model.tokenizer.eos_token,
            return_tensors='pt')
        # append the new user input tokens to the chat history
        bot_input_ids = torch.cat(
            [model.chat_history_ids,
             new_user_input_ids],
            device="GPU",
            dim=-1) if model.conversation_step > 0 else (
            new_user_input_ids)

        # generated a response while limiting the total
        # chat history to 1000 tokens,
        chat_history_ids = model.pipeline.generate(
            bot_input_ids, max_length=1000,
            pad_token_id=model.tokenizer.eos_token_id)
        print("Chatbot: {}".format(
            model.tokenizer.decode(
                chat_history_ids[:, bot_input_ids.shape[-1]:][0],
                skip_special_tokens=True)))
        print("User : How are you ?")
        new_user_input_ids = model.tokenizer.encode(
            "How are you ? " + model.tokenizer.eos_token,
            return_tensors='pt')
        bot_input_ids = torch.cat(
            [model.chat_history_ids, new_user_input_ids],
            dim=-1) if model.conversation_step > 0 else new_user_input_ids
        chat_history_ids = model.pipeline.generate(
            bot_input_ids,
            max_length=1000,
            pad_token_id=model.tokenizer.eos_token_id)

        print("Chatbot: {}".format(
            model.tokenizer.decode(
                chat_history_ids[:, bot_input_ids.shape[-1]:][0],
                skip_special_tokens=True)))
