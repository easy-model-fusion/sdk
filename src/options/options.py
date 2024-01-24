from abc import ABC


class Options(ABC):
    prompt: str
    device: str

    def __init__(self, prompt: str, device: str):
        self.prompt = prompt
        self.device = device
