from options import Options


class OptionsTextToImage(Options):
    image_width: int
    image_height: int

    def __init__(self, prompt: str, device: str, image_width: int, image_height: int):
        super().__init__(prompt, device)
        self.image_width = image_width
        self.image_height = image_height
