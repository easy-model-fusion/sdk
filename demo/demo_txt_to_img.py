from models.model_text_to_image import ModelTextToImage
from models.models_management import ModelsManagement

from options.options import Devices
from options.options_text_to_image import OptionsTextToImage


class DemoTxtToImg:

    def __init__(self):
        options = OptionsTextToImage(
            prompt="Astronaut in a jungle, cold color palette, "
                   "muted colors, detailed, 8k",
            device=Devices.GPU,
            image_width=512,
            image_height=512,
        )

        model_stabilityai_name = "stabilityai/sdxl-turbo"
        model_stabilityai_path = "stabilityai/sdxl-turbo"
        model_management = ModelsManagement()
        model_stabilityai = ModelTextToImage(model_stabilityai_name,
                                             model_stabilityai_path)

        model_management.add_model(new_model=model_stabilityai,
                                   model_options=options)
        model_management.load_model(model_stabilityai_name)

        image = model_management.generate_prompt()
        image.show()
