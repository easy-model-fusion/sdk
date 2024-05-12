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

    Attributes:
        tokenizer (Tokenizer): The tokenizer for the conversation model.
        model_pipeline (PreTrainedModel): The pretrained model
            for the conversation.
        tokenizer_pipeline (PreTrainedTokenizer): The pretrained
            tokenizer for the conversation.
        conversation (Conversation): The ongoing conversation instance.
        conversation_dict (Dict[uuid.UUID, Conversation]):
         A dictionary to store conversations with their IDs.
        schematic (Dict[str, str]): A dictionary representing the
            schematic of the conversation.
    """
    tokenizer: Tokenizer

    model_pipeline: PreTrainedModel
    tokenizer_pipeline: PreTrainedTokenizer
    conversation: Conversation

    conversation_dict: Dict[uuid.UUID, Conversation] = {}
    schematic: dict[str, str] = {"role": "user", "content": ""}

    def __init__(self, model_name: str,
                 model_path: str,
                 tokenizer_path: str,
                 model_class: Any,
                 tokenizer_class: Any,
                 device: Union[str, Devices]
                 ):
        """
        __init__ Initializes the Model conversational class.

        :param model_name: The name of the model.
        :param model_path: The path of the model.
        :param tokenizer_path: The path of the tokenizer.
        :param model_class: The model class use to interact
            with the model.
        :param tokenizer_class: The tokenizer class
            use to interact with the model.
        :param device: Which device the model must be on.
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
        generate_prompt Generates the prompt with the given option.

        :param prompt: The optional prompt.
        :param kwargs: Additional parameters for generating the prompt.

        :return: Generated prompt.
        """
        self.write_input(prompt)

        return self.transformers_pipeline(self.conversation)

    def write_input(self, prompt: str) -> None:
        """
        write_input Sends new input to the chatbot and generate a response.

        :param prompt: The input prompt for the chatbot.
        """
        self.schematic["content"] = prompt
        self.conversation.add_message(self.schematic)

    def create_new_conversation(self, **kwargs) -> None:
        """
        create_new_conversation Creates a new conversation.

        :param kwargs: Parameters for Conversation class.
        """
        conversation_uuid = uuid.uuid4()
        conversation = Conversation(conversation_id=conversation_uuid,
                                    **kwargs)
        self.conversation_dict[conversation_uuid] = conversation
        self.conversation = conversation

    def change_conversation(self, conversation_id: uuid.UUID) -> bool:
        """
        change_conversation Changes the active conversation.

        :param conversation_id: The ID of the conversation to switch to.

        :return: True if the conversation was switched successfully,
            False if the provided ID is invalid.
        """
        if conversation_id in self.conversation_dict:
            print("switching to conversation {}".format(conversation_id))
            self.conversation = self.conversation_dict[conversation_id]
            return True
        else:
            return False
