from src.options.options import Options, Devices


class OptionsTextToImage(Options):
    """
    Options for text-to-image models
    """
    prompt: str
    image_width: int
    image_height: int

    def __init__(self, device: Devices, prompt: str, image_width: int, image_height: int):
        """
        Initializes the OptionsTextToImage
        :param device: The device to use generate prompt
        :param prompt: The prompt to give to the model
        :param image_width: The width of the resulting image
        :param image_height: The height of the resulting image
        """
        super().__init__(device)
        self.prompt = prompt
        self.image_width = image_width
        self.image_height = image_height
