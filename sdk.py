from diffusers import DiffusionPipeline, StableDiffusionXLPipeline
import torch


class Text2ImgModel:
    pipeline: StableDiffusionXLPipeline
    model_name: str
    loaded: bool

    def __init__(self, model_name: str):
        self.loaded = False
        self.model_name = model_name
        self.create_pipeline()

    def create_pipeline(self):
        if self.loaded:
            return

        self.pipeline = DiffusionPipeline.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            use_safetensors=True,
            add_watermarker=False,
            variant="fp16",
        )

    def load_to(self, device: str):
        if self.loaded:
            return
        self.pipeline.to(device)
        self.loaded = True

    def unload(self):
        if not self.loaded:
            return
        self.pipeline.to(device="meta")
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        self.loaded = False

    def gen(self, prompt: str, width: int, height: int):
        return self.pipeline(prompt=prompt, width=width, height=height).images[0]


class Models:
    def __init__(self):
        self.loaded_model = None
        self.loaded_models_cache = {}

    def add_model(self, model_name: str):
        if model_name in self.loaded_models_cache:
            print(f"Model '{model_name}' is already in the cache.")
            return

        new_model = Text2ImgModel(model_name)
        self.loaded_models_cache[model_name] = new_model

    def load_model(self, model_name: str, device: str = "cuda"):
        if self.loaded_model:
            print("Unload the currently loaded model before loading a new one.")
            return

        if model_name not in self.loaded_models_cache:
            print(f"Model '{model_name}' cannot be loaded: not found.")
            return

        self.loaded_model = self.loaded_models_cache[model_name]
        self.loaded_model.load_to(device=device)

    def unload_model(self):
        if not self.loaded_model:
            print("No model loaded to unload.")
            return

        self.loaded_model.unload()
        self.loaded_model = None

    def generate_prompt(self, prompt: str, width: int, height: int):
        if not self.loaded_model:
            print("No model loaded. Load a model before generating prompts.")
            return

        return self.loaded_model.gen(prompt, width, height)

    def print_models(self):
        print("Models in cache:")
        for model_name, model_instance in self.loaded_models_cache.items():
            selected_indicator = "(selected)" if model_instance == self.loaded_model else ""
            print(f"- {model_name} {selected_indicator}")
