import torch
from typing import Optional, Dict
from transformers import (Conversation, AutoModelForCausalLM, AutoConfig)

from sdk.options.tokenizer_options import TokenizerOptions
from sdk.tokenizers.tokenizer_object import TokenizerObject
from sdk.models import Model
from sdk.options import Devices, OptionsTextConversation


class ModelsTextConversation(Model):
    pipeline: AutoModelForCausalLM
    tokenizer_object: TokenizerObject
    tokenizer_options: TokenizerOptions

    tokenizer_dict: Dict[int, TokenizerObject] = {}
    conversation_dict: Dict[int, Conversation] = {}

    current_conversation_id: int = 0
    current_tokenizer_id: int = 0

    conversation_ids: int = 0
    tokenizer_ids: int = 0

    loaded: bool

    chat_bot: Conversation
    conversation_step: int = 0

    # change this to dic
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
        self.pipeline = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            cache_dir="./cache",  # Cache directory
            return_dict=True,  # Return outputs as a dictionary
            pad_token_id=50256,  # Custom pad token ID
            eos_token_id=50256  # Custom end-of-sequence token ID
        )

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
        self.tokenizer_object = TokenizerObject(
            self.model_name,
            self.model_path,
            self.tokenizer_options
        )
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

    # Will be used to create new conversation
    def generate_prompt(
            self, prompt: Optional[str],
            option: OptionsTextConversation,
            **kwargs):
        prompt = prompt if prompt else option.prompt

        if option.create_new_conv:
            print("Create new chat")
            option.create_new_conv = False
            self.create_new_conversation(
                prompt=prompt,
                option=option
            )

        if option.create_new_tokenizer:
            print("Create new tokenizer")
            option.create_new_tokenizer = False
            self.create_new_tokenizer(
                tokenizer_options=self.tokenizer_options)

        if option.tokenizer_ID_to_use_id != self.current_tokenizer_id:
            print("Changing tokenizer")
            self.set_tokenizer_to_use(option.tokenizer_ID_to_use_id)

        if option.chat_ID_to_use_id != self.current_conversation_id:
            print("Changing chat id")
            self.change_conversation(option.chat_ID_to_use_id)

        if not self.conversation_active:
            self.current_conversation_id = self.conversation_ids
            print("setting conversation id to ", self.current_conversation_id)
            self.chat_bot = Conversation(prompt, **kwargs)
            # adding new conversation to dict
            self.conversation_dict[self.current_conversation_id] = (
                self.chat_bot
            )
            self.conversation_active = True
            self.conversation_ids += 1

        str_to_send = self.tokenizer_object.prepare_input(prompt)
        str_to_send = self.pipeline.generate(str_to_send,
                                             max_new_tokens=128)
        # print("BOT: ", self.model_name)
        return self.tokenizer_object.decode_model_output(str_to_send)

    def create_new_tokenizer(self,
                             tokenizer_options: TokenizerOptions):
        tokenizer = TokenizerObject(self.model_name,
                                    self.model_path,
                                    tokenizer_options)
        self.current_tokenizer_id = self.tokenizer_ids

        self.tokenizer_dict[self.current_tokenizer_id] = tokenizer
        self.tokenizer_ids += 1

    def set_tokenizer_to_use(self,
                             token_id: int):
        if token_id in self.tokenizer_dict:
            self.current_tokenizer_id = token_id
            self.tokenizer_object = self.tokenizer_dict[token_id]
        else:
            return -1

    def delete_tokenizer(self,
                         token_id: int):
        if token_id in self.tokenizer_dict:
            del self.tokenizer_dict[token_id]
        else:
            return -1

    def delete_conversation(self,
                            conversation_id: int):
        if conversation_id in self.conversation_dict:
            del self.conversation_dict[conversation_id]
        else:
            return -1

    def send_new_input(self,
                       prompt: str
                       ):
        input_ids = self.tokenizer_object.prepare_input(prompt)
        return self.pipeline.generate(**input_ids,
                                      max_new_tokens=128
                                      )

    def create_new_conversation(self,
                                prompt,
                                option: OptionsTextConversation,
                                **kwargs):

        self.current_conversation_id = self.conversation_ids
        self.chat_bot = Conversation(prompt, **kwargs)
        # adding new conversation to dict
        self.conversation_dict[self.current_conversation_id] = (
            self.chat_bot
        )
        self.conversation_ids += 1

    def change_conversation(self,
                            conversation_id: int):
        if conversation_id in self.conversation_dict:
            self.current_conversation_id = conversation_id
            self.chat_bot = self.conversation_dict[conversation_id]
        else:
            return -1
