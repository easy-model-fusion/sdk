from typing import Optional, Dict, Any

from sdk.models import Model
from sdk.options import Devices


class ModelsManagement:
    """
    The ModelsManagement class controls all instantiated models.
    It is with this class that you can deploy a model on a device and
    generate a prompt.
    """

    def __init__(self):
        """
        Initializes the ModelsManagement.
        """
        self.loaded_model_CPU: Optional[Model] = None
        self.loaded_model_GPU: Optional[Model] = None
        self.loaded_models_cache: Dict[str, Model] = {}

    def add_model(self, new_model: Model) -> bool:
        """
        Adds a new model and its options to the management.

        Args:
             new_model (Model): The new model to add.

        Returns:
            bool: True if the model is successfully added.
        """
        if new_model.model_name in self.loaded_models_cache:
            print(f"Model '{new_model.model_name}' is already in the cache.")
            return False

        self.loaded_models_cache[new_model.model_name] = new_model
        return True

    def load_model(self, model_name: str) -> bool:
        """
        Load a model with its name and the device set from the model option.

        Args:
             model_name (str): The name of the model to load.

        Returns:
            bool: True if the model is successfully loaded.
        """
        if model_name not in self.loaded_models_cache:
            print(f"Model '{model_name}' cannot be loaded: not found.")
            return False

        if (self.loaded_models_cache[model_name].device == Devices.CPU
                or self.loaded_models_cache[model_name].device == (
                        Devices.CPU.value)):

            return self.load_model_on_cpu(model_name)

        if (self.loaded_models_cache[model_name].device == Devices.GPU
                or self.loaded_models_cache[model_name].device == (
                        Devices.GPU.value)):

            return self.load_model_on_gpu(model_name)

    def load_model_on_cpu(self, model_name: str) -> bool:
        """
        Load a model with its name on the CPU.

        Args:
             model_name (str): The name of the model to load.

        Returns:
            bool: True if the model is successfully loaded.
        """
        if self.loaded_model_CPU:
            print(
                "Unload the currently loaded model before loading a new one.")
            return False

        self.loaded_model_CPU = self.loaded_models_cache[model_name]

        if not self.loaded_model_CPU.load_model():
            print("Something went wrong while unloading the model.")
            self.loaded_model_CPU = None
            return False

        return True

    def load_model_on_gpu(self, model_name: str) -> bool:
        """
        Load a model with its name on the GPU.

        Args:
             model_name (str): The name of the model to load.

        Returns:
            bool: True if the model is successfully loaded.
        """
        if self.loaded_model_GPU:
            print(
                "Unload the currently loaded model before loading a new one.")
            return False

        self.loaded_model_GPU = self.loaded_models_cache[model_name]

        if not self.loaded_model_GPU.load_model():
            print("Something went wrong while unloading the model.")
            self.loaded_model_GPU = None
            return False

        return True

    def unload_model(self, model_name: str) -> bool:
        """
        Unload the loaded model.

        Args:
             model_name (str): The name of the model to load.

        Returns:
            bool: True if the model is successfully unloaded.
        """
        if (self.loaded_models_cache[model_name].device == Devices.CPU
                or self.loaded_models_cache[model_name].device == (
                        Devices.CPU.value)):

            if not self.loaded_model_CPU:
                print("No model loaded to unload.")
                return False

            if not self.loaded_model_CPU.unload_model():
                print("Something went wrong while unloading the model.")
                return False
            self.loaded_model_CPU = None

        if (self.loaded_models_cache[model_name].device == Devices.GPU
                or self.loaded_models_cache[model_name].device == (
                        Devices.GPU.value)):

            if not self.loaded_model_GPU:
                print("No model loaded to unload.")
                return False

            if not self.loaded_model_GPU.unload_model():
                print("Something went wrong while unloading the model.")
                return False
            self.loaded_model_GPU = None

        return True

    def generate_prompt(self, prompt: Any,
                        model_name: str, **kwargs):
        """
        Generates the prompt for the loaded model with its stored options.

        Args:
            prompt (Any): The prompt to generate.
            model_name (str): The model name to load.
            kwargs: Additional parameters to pass to the prompt generator.

        Returns:
            The object of type link with the model category.
        """
        if (self.loaded_models_cache[model_name].device == Devices.CPU
                or self.loaded_models_cache[model_name].device == (
                        Devices.CPU.value)):

            if self.loaded_model_CPU.model_name != model_name:
                self.unload_model(model_name)

            if not self.loaded_model_CPU:
                self.load_model(model_name=model_name)

            return (
                self.loaded_model_CPU.generate_prompt(
                    prompt,
                    **kwargs
                )
            )

        if (self.loaded_models_cache[model_name].device == Devices.GPU
                or self.loaded_models_cache[model_name].device == (
                        Devices.GPU.value)):

            if self.loaded_model_GPU.model_name != model_name:
                self.unload_model(model_name)

            if not self.loaded_model_GPU:
                self.load_model(model_name=model_name)

            return (
                self.loaded_model_GPU.generate_prompt(
                    prompt,
                    **kwargs
                )
            )

    def print_models(self):
        """
        Prints all models in the cache.
        """
        print("Models in cache:")
        for model_name, model_instance in self.loaded_models_cache.items():
            selected_indicator_CPU = (
                "(CPU selected)" if model_instance == self.loaded_model_CPU
                else "")
            selected_indicator_GPU = (
                "(GPU selected)" if model_instance == self.loaded_model_GPU
                else "")
            print(f"- {model_name} {selected_indicator_CPU} "
                  f"{selected_indicator_GPU}")
