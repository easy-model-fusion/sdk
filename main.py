from src.models.models_management import ModelsManagement
from src.models.models_text_to_image import ModelsTextToImage
from src.options.options_text_to_image import OptionsTextToImage, Devices


if __name__ == '__main__':
    options = OptionsTextToImage(
        prompt="Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
        device=Devices.GPU,
        image_width=1024,
        image_height=1024
    )

    model_stabilityai_name = "stabilityai/sdxl-turbo"
    model_management = ModelsManagement()
    model_stabilityai = ModelsTextToImage(model_stabilityai_name)

    model_management.add_model(new_model=model_stabilityai, model_options=options)
    model_management.load_model(model_stabilityai_name)

    image = model_management.generate_prompt()
    image.show()

