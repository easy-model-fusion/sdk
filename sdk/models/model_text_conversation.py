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
        Initializes the Model conversational class

        Args:
            model_name (str): The name of the model
            model_path (str): The path of the model
            tokenizer_path (str): The path of the tokenizer
            model_class  (Any): The model class use to interact with the model
            tokenizer_class (Any):
                The tokenizer class use to interact with the model
            device (Union[str, Devices]): Which device the model must be on
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
        """
        self.schematic["content"] = prompt
        self.conversation.add_message(self.schematic)

    def create_new_conversation(self, **kwargs) -> None:
        """
        Create a new conversation.

        Args:
             kwargs: parameters for Conversation class
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
            conversation_id (UUID): The ID of the conversation to switch to.

        Returns:
            bool: True if the conversation was switched successfully,
                False if the provided ID is invalid.
        """
        if conversation_id in self.conversation_dict:
            print("switching to conversation {}".format(conversation_id))
            self.conversation = self.conversation_dict[conversation_id]
            return True
        else:
            return False
