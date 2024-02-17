from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from options.options_text_conversation import OptionsTextConversation
from src.models.models_management import ModelsManagement
from src.models.models_text_conversation import ModelsTextConversation
from src.options.options import Devices

model_stabilityai_name = "facebook/blenderbot-400M-distill"
if __name__ == '__main__':
    options = OptionsTextConversation(
        prompt="Astronaut in a Fur suit in an anime convention, disgust",
        device=Devices.CPU,
        model=model_stabilityai_name,
        batch_size=3,
        minimum_tokens=50
    )

    model_management = ModelsManagement()
    model_stabilityai = ModelsTextConversation(model_stabilityai_name)

    model_management.add_model(new_model=model_stabilityai, model_options=options)
    model_management.load_model(model_stabilityai_name)

    model = ModelsTextConversation(model_stabilityai_name)

    model.create_pipeline()
    model.generate_prompt(options)
