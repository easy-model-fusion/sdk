import torch
from typing import Optional, Union
from transformers import (AutoModel, AutoTokenizer,
                          ConversationalPipeline, Conversation)

from sdk.options.tokenizer_options import TokenizerOptions
from sdk.tokenizers.tokenizer_object import TokenizerObject
from sdk.models import Model
from sdk.options import Devices, OptionsTextConversation


class ModelsTextConversation(Model):
    pipeline: ConversationalPipeline
    tokenizer_object: TokenizerObject
    tokenizer_options: TokenizerOptions
    loaded: bool
    chat_id: int = 0
    chat_bot: Conversation
    conversation_step: int = 0
    chat_history_token_ids = []
    conversation_active: bool = False

    def __init__(self, model_name: str, model_path: str):
        """
        Initializes the ModelsTextToImage class
        :param model_name: The name of the model
        :param model_path: The path of the model
        """
        super().__init__(model_name, model_path)
        self.loaded = False
        self.create_pipeline()

    def create_pipeline(self):
        """
        Creates the pipeline to load on the device
        """
        if self.loaded:
            return
        self.pipeline = AutoModel.from_pretrained(self.model_path)

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
        self.tokenizer_object = TokenizerObject(self.model_name, self.model_path)
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

    def generate_prompt(
            self, prompt: Optional[str],
            option: OptionsTextConversation,
            **kwargs):
        prompt = prompt if prompt else option.prompt
        self.start_conversation(prompt=prompt)

    def start_conversation(self, prompt: Optional[str],
                           **kwargs):
        prompt = prompt if prompt else "Hello"
        print("Model name : ", self.model_name)

        """ Here init tokenizer object"""

        self.chat_bot = Conversation(prompt, **kwargs)

        # How to go into conversation loop here, and get out of it
        while self.conversation_active:
            # Get user input
            user_input = input("You: ")
            # Tokenize user input
            inputs = self.tokenizer_object.tokenizer.encode(
                prompt=user_input +
                       self.tokenizer_object.tokenizer.eos_token,
                return_tensors="pt")

            bot_input_ids = torch.cat(
                [self.pipeline.chat_history_ids,
                 inputs],
                device="GPU",
                dim=-1)
            chat_history_ids = self.pipeline.generate(
                bot_input_ids, max_length=1000,
                pad_token_id=self.tokenizer_object.tokenizer.eos_token_id)
            print("Chatbot: {}".format(
                self.tokenizer_object.tokenizer.decode(
                    chat_history_ids[:, bot_input_ids.shape[-1]:][0],
                    skip_special_tokens=True)))
