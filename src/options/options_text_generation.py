from src.options.options import Options


class OptionsTextGeneration(Options):
    max_length: int
    temperature: float

    def __init__(self, prompt: str, device: str, max_length: int, temperature: float):
        super().__init__(prompt, device)
        self.max_length = max_length
        self.temperature = temperature
