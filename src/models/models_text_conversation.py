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

    def __init__(self, model_name: str):
        """
        Initializes the ModelsTextToImage class
        :param model_name: The name of    chat_bot: Conversation
 the model
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

    def generate_prompt(self, option: OptionsTextConversation):
        Conversation(option.prompt)
