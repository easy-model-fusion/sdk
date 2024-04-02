from typing import Optional, Dict, Any
from sdk.models import Model


class ModelsManagement:
    """
    The ModelsManagement class controls all instantiated models.
    It is with this class that you can deploy a model on a device and
    generate a prompt.
    """

    def __init__(self):
        """
        Initializes the ModelsManagement
        """
        self.loaded_model: Optional[Model] = None
        self.loaded_models_cache: Dict[str, Model] = {}

    def add_model(self, new_model: Model) -> bool:
        """
        Adds a new model and his options to the management.

        Args:
             new_model (Model): The new model to add

        Returns:
            bool: True if the model is successfully added
        """
        if new_model.model_name in self.loaded_models_cache:
            print(f"Model '{new_model.model_name}' is already in the cache.")
            return False

        self.loaded_models_cache[new_model.model_name] = new_model
        return True

    def load_model(self, model_name: str) -> bool:
        """
        Load a model with his name and the device set from de model option.

        Args:
             model_name (str): The name of the model to load

        Returns:
            bool: True if the model is successfully loaded
        """
        if self.loaded_model:
            print(
                "Unload the currently loaded model before loading a new one.")
            return False

        if model_name not in self.loaded_models_cache:
            print(f"Model '{model_name}' cannot be loaded: not found.")
            return False

        self.loaded_model = self.loaded_models_cache[model_name]
        if not self.loaded_model.load_model():
            print("Something went wrong while unloading the model.")
            self.loaded_model = None
            return False

        return True

    def unload_model(self) -> bool:
        """
        Unload the loaded model

        Returns:
            bool: True if the model is successfully unloaded
        """
        if not self.loaded_model:
            print("No model loaded to unload.")
            return False

        if not self.loaded_model.unload_model():
            print("Something went wrong while unloading the model.")
            return False
        self.loaded_model = None
        return True

    def generate_prompt(self, prompt: Any,
                        model_name: Optional[str] = None, **kwargs):
        """
        Generates the prompt for the loaded model with his stored options

        Args:
            prompt (Any): The prompt to generate
            model_name: (Optional[str]): the model name to load
            kwargs: more parameters to pass to the prompt generator

        Returns:
            The object of type link with the model category
        """
        if model_name:
            if self.loaded_model.model_name != model_name:
                self.unload_model()

        if not self.loaded_model:
            if model_name:
                print("No model loaded to generate.")
                return

            self.load_model(model_name=model_name)

        return (
            self.loaded_model.generate_prompt(
                prompt,
                **kwargs
            )
        )

    def print_models(self):
        """
        Prints all models in the cache
        """
        print("Models in cache:")
        for model_name, model_instance in self.loaded_models_cache.items():
            selected_indicator = (
                "(selected)" if model_instance == self.loaded_model else "")
            print(f"- {model_name} {selected_indicator}")
