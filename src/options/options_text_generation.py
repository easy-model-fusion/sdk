from src.options.options import Options, Devices


class OptionsTextGeneration(Options):
    """
    Options for text-Generation models
    """
    prompt: str
    max_length: int
    temperature: float

    def __init__(self, prompt: str, device: Devices, max_length: int, temperature: float):
        """
        Initializes the OptionsTextGeneration
        :param device: The device to use generate prompt
        :param prompt: The prompt to give to the model
        :param max_length: The max length of the generated response
        :param temperature: parameter used during the
         sampling process to control the randomness of generated text
         High temp : High randomness, Low temp : Low randomness
        """
        super().__init__(device)
        self.prompt = prompt
        self.max_length = max_length
        self.temperature = temperature
