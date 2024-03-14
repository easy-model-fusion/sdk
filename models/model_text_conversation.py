import torch
from typing import Optional, Dict, Tuple
from transformers import (Conversation, AutoModelForCausalLM, AutoConfig)

from sdk.options.options_tokenizer import OptionsTokenizer
from sdk.tokenizers.tokenizer_object import TokenizerObject
from sdk.models import Model
from sdk.options import Devices, OptionsTextConversation


class ModelsTextConversation(Model):
    """
    A class representing a text conversation model.

    Attributes:
        pipeline (AutoModelForCausalLM): The pipeline for the text conversation.
        tokenizer_object (TokenizerObject): The tokenizer object for the model.
        tokenizer_options (OptionsTokenizer): The options for the tokenizer.
        tokenizer_dict (Dict[int, TokenizerObject]): Dictionary to store tokenizers.
        conversation_dict (Dict[int, Conversation]): Dictionary to store conversations.
        current_conversation_id (int): ID of the current conversation.
        current_tokenizer_id (int): ID of the current tokenizer.
        conversation_ctr (int): Total number of conversations.
        tokenizer_ctr (int): Total number of tokenizers.
        loaded (bool): Flag indicating whether the model is loaded.
        chat_bot (Conversation): Current conversation.
        conversation_step (int): Step of the conversation.
        conversation_history (list): List to store chat history token IDs.
        conversation_active (bool): Flag indicating if a conversation is active.
    """
    pipeline: AutoModelForCausalLM
    tokenizer_object: TokenizerObject
    tokenizer_options: OptionsTokenizer

    tokenizer_dict: Dict[int, TokenizerObject] = {}
    conversation_dict: Dict[int, Tuple[Conversation, list]] = {}

    current_conversation_id: int = 0
    current_tokenizer_id: int = 0

    conversation_ctr: int = 0
    tokenizer_ctr: int = 0

    loaded: bool

    chat_bot: Conversation = None
    conversation_step: int = 0

    # change this to dic
    conversation_history = []
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
            self.model_path,
            trust_remote_code=True,
            cache_dir="./cache",  # Cache directory
            return_dict=True,  # Return outputs as a dictionary
            pad_token_id=50256,  # Custom pad token ID
            eos_token_id=50256  # Custom end-of-sequence token ID
        )

    def load_model(self, option: OptionsTextConversation) -> bool:
        """
        Load this model on the given device.

        Args:
            option (OptionsTextConversation): The options with the device.

        Returns:
            bool: True if the model is successfully loaded.
        """

        if self.loaded:
            return True
        if option.device == Devices.RESET:
            return False

        self.create_new_tokenizer(tokenizer_options=self.tokenizer_options)
        self.pipeline.to(option.device.value)
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
        """
        Generates the prompt with the given option.

        Args:
            prompt (Optional[str]): The optional prompt.
            option (OptionsTextConversation): The options of text conversation model.

        Returns:
            str: Generated prompt.
        """
        prompt = prompt if prompt else option.prompt

        if option.create_new_conv:
            self.create_new_conversation(
                prompt=prompt,
                option=option
            )
        if option.chat_id_to_use != self.current_conversation_id:
            self.change_conversation(option.chat_id_to_use)

        if option.create_new_tokenizer:
            print("creating tokenizer")
            option.create_new_tokenizer = False
            self.create_new_tokenizer(
                tokenizer_options=self.tokenizer_options)

        if option.tokenizer_id_to_use != self.current_tokenizer_id:
            print("changing tokenizer")
            self.set_tokenizer_to_use(option.tokenizer_id_to_use)

        if not self.conversation_active:
            print("creating conv")
            self.create_new_conversation(
                prompt=prompt,
                option=option
            )

        history = '\n'.join(self.conversation_history)
        print("sending input")
        str_to_send = self.tokenizer_object.prepare_input(prompt, history)
        result = self.pipeline.generate(
            str_to_send,
            max_new_tokens=128
        )
        response = self.tokenizer_object.decode_model_output(result)

        self.conversation_history.append(prompt)
        self.conversation_history.append(response)
        return response

    """
    Creates a new tokenizer.

    Args:
        tokenizer_options (TokenizerOptions): Options for the tokenizer.
    """

    def create_new_tokenizer(self,
                             tokenizer_options: OptionsTokenizer) -> int:

        tokenizer = TokenizerObject(self.model_name,
                                    self.model_path,
                                    tokenizer_options)
        self.current_tokenizer_id = self.tokenizer_ctr
        self.tokenizer_dict[self.current_tokenizer_id] = tokenizer
        self.tokenizer_object = tokenizer
        self.tokenizer_ctr += 1
        return self.current_tokenizer_id

    def set_tokenizer_to_use(self, token_id: int) -> bool:
        """
        Set the tokenizer to use for generating text.

        Args:
            token_id (int): The ID of the tokenizer to use.

        Returns:
            int: The ID of the selected tokenizer, or -1 if the provided ID is invalid.
        """
        if token_id in self.tokenizer_dict:
            self.current_tokenizer_id = token_id
            self.tokenizer_object = self.tokenizer_dict[token_id]
            return True
        else:
            return False

    def delete_tokenizer(self, token_id: int) -> bool:
        """
        Delete a tokenizer.

        Args:
            token_id (int): The ID of the tokenizer to delete.

        Returns:
            int: 0 if the tokenizer was deleted successfully, -1 if the provided ID is invalid.
        """
        if token_id in self.tokenizer_dict:
            del self.tokenizer_dict[token_id]
            return True
        else:
            return False

    def delete_conversation(self, conversation_id: int) -> bool:
        """
        Delete a conversation.

        Args:
            conversation_id (int): The ID of the conversation to delete.

        Returns:
            int: 0 if the conversation was deleted successfully, -1 if the provided ID is invalid.
        """
        if conversation_id in self.conversation_dict:
            del self.conversation_dict[conversation_id]
            if len(self.conversation_dict) == 0:
                self.conversation_active = False
            return True
        else:
            return False

    def send_new_input(self, prompt: str, history: list) -> str:
        """
        Send new input to the chatbot and generate a response.

        Args:
            history: Chat history
            prompt (str): The input prompt for the chatbot.

        Returns:
            str: The generated response from the chatbot.
        """
        input_ids = {"input_ids": self.tokenizer_object.prepare_input(prompt, history=history)}
        return self.pipeline.generate(**input_ids)

    def create_new_conversation(self, prompt,
                                option: OptionsTextConversation, **kwargs) -> int:
        """
        Create a new conversation.

        Args:
            prompt: The initial prompt for the conversation.
            option (OptionsTextConversation): The options for the conversation.
            **kwargs: Additional keyword arguments for initializing the conversation.

        Returns:
            None
        """
        option.create_new_conv = False
        option.chat_id_to_use = self.conversation_ctr
        self.conversation_active = True

        if self.chat_bot is not None:
            self.save_conversation()

        # create new conversation and increment counter
        self.current_conversation_id = self.conversation_ctr
        self.chat_bot = Conversation(prompt, **kwargs)

        # clearing conversation history
        self.conversation_history = []
        # adding new conversation to dict, with cleared history
        self.conversation_dict[self.current_conversation_id] = self.chat_bot, []
        # incrementing counter
        self.conversation_ctr += 1
        return self.current_conversation_id

    def change_conversation(self, conversation_id: int) -> bool:
        """
        Change the active conversation.

        Args:
            conversation_id (int): The ID of the conversation to switch to.

        Returns:
            int: 0 if the conversation was switched successfully, -1 if the provided ID is invalid.
        """
        if conversation_id in self.conversation_dict:
            print("switching to conversation {}".format(conversation_id))
            self.current_conversation_id = conversation_id
            self.chat_bot, self.conversation_history = self.conversation_dict[conversation_id]
            return True
        else:
            return False

    def save_conversation(self) -> None:
        # save state of  conversation in dict
        self.conversation_dict[self.current_conversation_id] = self.chat_bot, self.conversation_history
