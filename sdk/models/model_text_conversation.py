import uuid
from typing import Dict, Union, Any
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    Conversation
)

from sdk.tokenizers.tokenizer import Tokenizer
from sdk.models import ModelTransformers
from sdk.options import Devices


class ModelsTextConversation(ModelTransformers):
    """
    A class representing a text conversation model.
    """
    tokenizer: Tokenizer

    model_pipeline: PreTrainedModel
    tokenizer_pipeline: PreTrainedTokenizer
    conversation: Conversation

    conversation_dict: Dict[uuid.UUID, Conversation] = {}
    schematic: dict[str, str] = {"role": "user", "content": ""}

    def __init__(self, model_name: str,
                 model_path: str,
                 tokenizer_path,
                 model_class: Any,
                 tokenizer_class: Any,
                 device: Union[str, Devices]
                 ):
        """
        Initializes the ModelsTextToImage class
        :param model_name: The name of the model
        :param model_path: The path of the model
        :param device: Which device the model must be on
        """
        super().__init__(model_name=model_name,
                         model_path=model_path,
                         tokenizer_path=tokenizer_path,
                         task="conversational",
                         model_class=model_class,
                         tokenizer_class=tokenizer_class,
                         device=device)

    def generate_prompt(
            self, prompt: str,
            **kwargs) -> Union[Conversation, None]:
        """
        Generates the prompt with the given option.

        Args:
            prompt: (str): The optional prompt.

        Returns:
            str: Generated prompt.
        """
        self.write_input(prompt)
        return self.transformers_pipeline(self.conversation)

    def write_input(self, prompt: str) -> None:
        """
        Send new input to the chatbot and generate a response.

        Args:
            prompt (str): The input prompt for the chatbot.

        Returns:
            str: The generated response from the chatbot.
        """
        self.schematic["content"] = prompt

        self.conversation.add_message(self.schematic)

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
