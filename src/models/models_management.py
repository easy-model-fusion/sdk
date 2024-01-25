from typing import Optional, Dict
from src.models.models import Models
from src.options.options import Options


class ModelsManagement:
    """
    The ModelsManagement class controls all instantiated models.
    It is with this class that you can deploy a model on a device and generate a prompt.
    """

    def __init__(self):
        """
        Initializes the ModelsManagement
        """
        self.loaded_model: Optional[Models] = None
        self.loaded_models_cache: Dict[str, Models] = {}
        self.options_models: Dict[str, Options] = {}

    def add_model(self, new_model: Models, model_options: Options):
        """
        Adds a new model and his options to the management.
        :param new_model: The new model to add
        :param model_options: The options of the new model to add
        """
        if new_model.model_name in self.loaded_models_cache:
            print(f"Model '{new_model.model_name}' is already in the cache.")
            return

        self.loaded_models_cache[new_model.model_name] = new_model
        self.options_models[new_model.model_name] = model_options

    def load_model(self, model_name: str):
        """
        Load a model with his name and the device set from de model option.
        :param model_name: The name of the model to load
        """
        if self.loaded_model:
            print("Unload the currently loaded model before loading a new one.")
            return

        if model_name not in self.loaded_models_cache:
            print(f"Model '{model_name}' cannot be loaded: not found.")
            return

        self.loaded_model = self.loaded_models_cache[model_name]
        self.loaded_model.load_model(option=self.options_models[model_name])

    def unload_model(self):
        """
        Unload the loaded model
        """
        if not self.loaded_model:
            print("No model loaded to unload.")
            return

        self.loaded_model.unload_model()
        self.loaded_model = None

    def get_model_options(self, model_name: str) -> Options:
        """
        Gets the options of the model with the given name
        :param model_name: The name of a model
        :return: The object options of the model
        """
        return self.options_models[model_name]

    def set_model_options(self, model_name: str, options: Options):
        """
        Set the options of the model with the given name
        :param model_name: The name of a model
        :param options: The object options of the model
        """
        self.options_models[model_name] = options

    def generate_prompt(self):
        """
        Generates the prompt for the loaded model with his stored options
        :return: The object of type link with the model category
        """
        if not self.loaded_model:
            print("No model loaded. Load a model before generating prompts.")
            return

        return self.loaded_model.generate_prompt(self.options_models[self.loaded_model.model_name])

    def print_models(self):
        """
        Prints all models in the cache
        """
        print("Models in cache:")
        for model_name, model_instance in self.loaded_models_cache.items():
            selected_indicator = "(selected)" if model_instance == self.loaded_model else ""
            print(f"- {model_name} {selected_indicator}")
