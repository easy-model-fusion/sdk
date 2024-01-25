from src.models.models_management import ModelsManagement
from src.models.models_text_generation import ModelsTextGeneration
from src.models.models_text_to_image import ModelsTextToImage
from src.options.options_text_generation import OptionsTextGeneration
from src.options.options_text_to_image import OptionsTextToImage, Devices

if __name__ == '__main__':
    options = OptionsTextGeneration(
        prompt="Astronaut in a Fur suit in an anime convention, disgust",
        max_length=200,
        temperature=0.2,
        device=Devices.CPU
    )

    model_stabilityai_name = "stabilityai/stable-code-3b"
    model_management = ModelsManagement()
    model_stabilityai = ModelsTextGeneration(model_stabilityai_name)

    model_management.add_model(new_model=model_stabilityai, model_options=options)
    model_management.load_model(model_stabilityai_name)


    text = model_management.generate_prompt()
    print(text)
