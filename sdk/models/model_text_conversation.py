import torch
import uuid
from typing import Optional, Dict, Union
from transformers import (
    pipeline,
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizer,
    Conversation
)
from sdk.tokenizers.tokenizer import Tokenizer
from sdk.models import Model
from sdk.options import Devices


class ModelsTextConversation(Model):
    """
    A class representing a text conversation model.

    Attributes:
        pipeline (AutoModelForCausalLM):
            The pipeline for the text conversation.
        tokenizer_object (Tokenizer):
            The tokenizer object for the model.
        tokenizer_options (OptionsTokenizer):
            The options for the tokenizer.
        tokenizer_dict (Dict[int, TokenizerObject]):
            Dictionary to store tokenizers.
        conversation_dict (Dict[int, Tuple[Conversation, list]]):
            Dictionary to store conversations.
        current_conversation_id (int):
            ID of the current conversation.
        current_tokenizer_id (int): ID of the current tokenizer.
        conversation_ctr (int): Total number of conversations.
        tokenizer_ctr (int): Total number of tokenizers.
        loaded (bool): Flag indicating whether the model is loaded.
        conversation_step (int): Step of the conversation.
        conversation_history (list):
            List to store chat history token IDs.
        conversation_active (bool):
         Flag indicating if a conversation is active.
    """
    tokenizer: Tokenizer
    device: Union[str, Devices]

    conversation_pipeline: pipeline
    model_pipeline: PreTrainedModel
    tokenizer_pipeline: PreTrainedTokenizer
    conversation: Conversation

    conversation_dict: Dict[uuid.UUID, Conversation] = {}

    loaded: bool

    def __init__(self, model_name: str, model_path: str,
                 tokenizer: Tokenizer,
                 device: Union[str, Devices]
                 ):
        """
        Initializes the ModelsTextToImage class
        :param model_name: The name of the model
        :param model_path: The path of the model
        :param device: Which device the model must be on
        """
        super().__init__(model_name, model_path)
        self.device = device
        self.loaded = False
        self.tokenizer = tokenizer
        self.create_pipeline()

    def create_pipeline(self, **kwargs) -> None:
        """
        Creates the pipeline to load on the device
        """
        if self.loaded:
            return

        self.model_pipeline = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            **kwargs
        )
        if not self.tokenizer or not self.tokenizer.pipeline:
            print("The tokenizer pipeline is required to load the model")
            return

        self.tokenizer_pipeline = self.tokenizer.pipeline
        self.conversation_pipeline = pipeline(
            "conversational",
            model=self.model_pipeline,
            tokenizer=self.tokenizer_pipeline,
            device=(
                self.device if isinstance(
                    self.device, str) else (
                    self.device.value))
        )

    def load_model(self) -> bool:
        """
        Load this model on the given device.

        Args:
            option (OptionsTextConversation): The options with the device.

        Returns:
            bool: True if the model is successfully loaded.
        """

        if self.loaded:
            return True
        if self.device == Devices.RESET.value:
            return False
        self.conversation_pipeline.to(device=(
            self.device if isinstance(
                self.device, str) else (
                self.device.value)))
        self.loaded = True
        return True

    def unload_model(self) -> bool:
        """
        Unloads the model
        :return: True if the model is successfully unloaded
        """
        if not self.loaded:
            return False

        self.conversation_pipeline.to(device=(
            self.device if isinstance(
                self.device, str) else (
                self.device.value))
        )
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        self.loaded = False
        return True

    # Will be used to create new conversation
    def generate_prompt(
            self, prompt: str,
            **kwargs):
        """
        Generates the prompt with the given option.

        Args:
            prompt (Optional[str]): The optional prompt.

        Returns:
            str: Generated prompt.
        """
        if not self.conversation:
            return None

        self.write_input(prompt)
        return self.conversation_pipeline(self.conversation)

    def write_input(self, prompt: str) -> None:
        """
        Send new input to the chatbot and generate a response.

        Args:
            prompt (str): The input prompt for the chatbot.

        Returns:
            str: The generated response from the chatbot.
        """
        # ToDo
        schematic = ({"role": "user", "content": prompt})
        self.conversation.add_message(schematic)

    def create_new_conversation(self, **kwargs) -> None:
        """
        Create a new conversation.
        """
        conversation_uuid = uuid.uuid4()
        conversation = Conversation(conversation_id=conversation_uuid,
                                    **kwargs)
        self.conversation_dict[conversation_uuid] = conversation
        self.conversation = conversation

    def change_conversation(self, conversation_id: uuid.UUID) -> bool:
        """
        Change the active conversation.

        Args:
            conversation_id (int): The ID of the conversation to switch to.

        Returns:
            int: 0 if the conversation was switched successfully,
                -1 if the provided ID is invalid.
        """
        if conversation_id in self.conversation_dict:
            print("switching to conversation {}".format(conversation_id))
            self.conversation = self.conversation_dict[conversation_id]
            return True
        else:
            return False
