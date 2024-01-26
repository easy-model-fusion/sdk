from diffusers import DiffusionPipeline, StableDiffusionXLPipeline
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.models.models import Models
from src.options.options import Devices
from src.options.options_text_generation import OptionsTextGeneration


class ModelsTextGeneration(Models):
    pipeline: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    model_name: str
    loaded: bool

    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.loaded = False
        self.create_pipeline()

    def create_pipeline(self):
        if self.loaded:
            return

        #other options ?
        self.pipeline = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            trust_remote_code=True
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

    def load_model(self, option: OptionsTextGeneration) -> bool:
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
        self.loaded = True
        return True

    def unload_model(self):
        if not self.loaded:
            return
        self.pipeline.to(device="meta")
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        self.loaded = False

    def generate_prompt(self, option: OptionsTextGeneration):
        inputs = self.tokenizer(option.prompt, return_tensors="pt").to(option.device.value)
        
        return self.pipeline(**inputs)
