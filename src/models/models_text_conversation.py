from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, ConversationalPipeline, Conversation
from src.models.model import Model
from src.options.options import Devices
from src.options.options_text_conversation import OptionsTextConversation
from transformers import pipeline


class ModelsTextConversation(Model):
    pipeline: ConversationalPipeline
    tokenizer: AutoTokenizer
    model_name: str
    loaded: bool
    chat_bot: Conversation
    conversation_step: int = 0
    chat_history_token_ids = []

    def __init__(self, model_name: str):
        """
        Initializes the ModelsTextToImage class
        :param model_name: The name of the model
        """
        super().__init__(model_name)
        self.loaded = False
        self.create_pipeline()

    def create_pipeline(self):
        """
        Creates the pipeline to load on the device
        """
        if self.loaded:
            return

        self.pipeline = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True, padding_side='left')

    def load_model(self, option: OptionsTextConversation) -> bool:
        """
        Load this model on the given device
        :param option: The options with the device
        :return: True if the model is successfully loaded
        """
        if self.loaded:
            return True
        if option.device == Devices.RESET:
            return False
        self.pipeline.to(option.device.value)
        self.tokenizer
        self.loaded = True
        return True

    def unload_model(self):
        """
        Unloads the model
        :return: True if the model is successfully unloaded
        """
        if not self.loaded:
            return False
        self.pipeline.to(device=Devices.RESET)
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        self.loaded = False
        return True

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

    def generate_prompt(self,prompt: Optional[str], option: OptionsTextConversation):
        prompt=prompt if prompt else option.prompt
        Conversation(prompt)
        for step in range(option.max_steps):
            self.add_input()
