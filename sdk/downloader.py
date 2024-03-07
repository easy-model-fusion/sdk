import argparse
import json
import os
import sys
import importlib

import transformers

# Authorized module names for download
DIFFUSERS = 'diffusers'
TRANSFORMERS = 'transformers'
AUTHORIZED_MODULE_NAMES = {DIFFUSERS, TRANSFORMERS}

# Map module name to a default class name
model_config_default_class_for_module = {
    DIFFUSERS: "DiffusionPipeline",
    TRANSFORMERS: "AutoModel",
}

# Transformers defaults
TRANSFORMERS_DEFAULT_MODEL_DIRECTORY = "model"
TRANSFORMERS_DEFAULT_TOKENIZER_CLASS = "AutoTokenizer"

# Download possibilities
DOWNLOAD_MODEL = "model"
DOWNLOAD_TOKENIZER = "tokenizer"

# Skip arguments
SKIP_TAG = "--skip"
SKIP_ARGUMENTS = {DOWNLOAD_MODEL, DOWNLOAD_TOKENIZER}

# Error exit codes
ERROR_EXIT_DEFAULT = 1
ERROR_EXIT_MODEL = 2
ERROR_EXIT_TOKENIZER = 3


def exit_error(message, error=ERROR_EXIT_DEFAULT):
    """
    Print an error message to the standard error stream and exit the script
    with a specified error code.

    Args:
        message (str): The error message to be printed.
        error (int, optional): The error code to exit the script with.
            Defaults to 1.
    """
    print(message, file=sys.stderr)
    sys.exit(error)


class Tokenizer:
    """
    A class representing a Tokenizer.

    Attributes:
        download_path (str): Path to where the tokenizer will be downloaded.
        class_name (str): The class name of the tokenizer.
        options (list): List of options for the tokenizer.
    """

    def __init__(self, class_name: str = None, options: list = None):
        self.download_path = None
        self.class_name = class_name or TRANSFORMERS_DEFAULT_TOKENIZER_CLASS
        self.options = options or []


class Model:
    """
    A class representing a Model.

    Attributes:
        base_path (str): Path to the model directory.
        download_path (str): Path to where the model will be downloaded.
        name (str): The name of the model to download.
        module (str): The name of the module to use for downloading the model.
        class_name (str, optional): The class name within the module to use
            for downloading the model. Defaults to None.
        options (list, optional): List of options used for downloading the
            model. Defaults to None.
        access_token (str, optional): Access token for downloading the model.
        tokenizer (Tokenizer, optional): Tokenizer object for the model.
            Defaults to None.
    """

    def __init__(self, name: str, module: str, class_name: str = None,
                 options: list = None, access_token: str = None,
                 tokenizer: Tokenizer = None):
        self.base_path = None
        self.download_path = None
        self.name = name
        self.module = module
        self.class_name = class_name
        self.options = options or []
        self.access_token = access_token
        self.tokenizer = tokenizer

    def validate(self):
        """
        Validate the model.

        Returns:
            Program exits if invalid
        """

        # Check if the model name is not empty
        if not self.name.strip():
            exit_error(f"Model '{self.name}' is invalid.")

    def is_transformers(self) -> bool:
        """
        Check if the model belongs to the Transformers module.

        Returns:
            bool: True if the model belongs to Transformers, False otherwise.
        """
        return self.module == TRANSFORMERS

    def build_paths(self, models_path: str) -> None:
        """
        Build paths for the model.

        Args:
            models_path (str): The base path where all the models are located.

        Returns:
            None
        """

        # Local path to the model directory
        self.base_path = os.path.join(models_path, self.name)

        # Local path where the model will be downloaded
        self.download_path = self.base_path

        # Improved repartition required when using transformers
        if self.is_transformers():
            self.download_path = os.path.join(
                self.base_path, TRANSFORMERS_DEFAULT_MODEL_DIRECTORY)

    def download(self, models_path: str, skip: str = "",
                 overwrite=False) -> str:
        """
        Download the model.

        Args:
            models_path (str): The base path where all the models are located.
            skip (str): Optional. Skips the download process of either the
                model or the tokenizer.
            overwrite (bool): Optional. Whether to overwrite the downloaded
                model if it exists.

        Returns:
            Program exits with error if the download fails.
            If it succeeds, it returns the JSON props used for downloading
                the model.
        """

        # Validate mandatory arguments
        self.validate()

        # Build paths
        self.build_paths(models_path)

        # Output result
        result_dict = {}

        # Checking for model download
        if skip != DOWNLOAD_MODEL:

            # Downloading the model
            download_model(self, overwrite)

            # Adding properties to result
            result_dict["path"] = self.download_path
            result_dict["module"] = self.module
            result_dict["class"] = self.class_name

        # Checking for tokenizer download
        if self.is_transformers() and skip != DOWNLOAD_TOKENIZER:

            # Download a tokenizer for the model
            download_transformers_tokenizer(self, overwrite)

            # Adding properties to result
            result_dict["tokenizer"] = {
                "path": self.tokenizer.download_path,
                "class": self.tokenizer.class_name,
            }

        # Convert the dictionary to JSON
        return json.dumps(result_dict, indent=4)


def download_model(model: Model, overwrite: bool) -> None:
    """
    Download the model.

    Args:
        model (Model): Model to be downloaded.
        overwrite (bool): Whether to overwrite the downloaded model if
            it exists.

    Returns:
        None. Exit with error if anything goes wrong.
    """

    # Check if the model already exists at path
    if is_path_valid_for_download(model.download_path, overwrite):
        exit_error(f"Model '{model.download_path}' already exists.")

    # Model class is not provided, trying the default one
    if model.class_name is None or model.class_name.strip() == '':
        model.class_name = model_config_default_class_for_module.get(
            model.module)

    # Processing options
    options = process_options(model.options or [])

    try:
        # Transforming from strings to actual objects
        module_obj = globals()[model.module]
        model_class_obj = getattr(module_obj, model.class_name)

        # Downloading the model
        model_downloaded = model_class_obj.from_pretrained(
            model.name, **options)
        model_downloaded.save_pretrained(model.download_path)

    except Exception as e:
        exit_error(f"Error while downloading model {model.name}: {e}",
                   ERROR_EXIT_MODEL)


def download_transformers_tokenizer(model: Model, overwrite: bool) -> None:
    """
    Download a transformers tokenizer for the model.

    Args:
        model (Model): Model to be downloaded.
        overwrite (bool): Whether to overwrite the downloaded model if
            it exists.

    Returns:
        None. Exit with error if anything goes wrong.
    """

    try:

        # Retrieving tokenizer class from module
        tokenizer_class_obj = getattr(transformers, model.tokenizer.class_name)

        # Local path where the tokenizer will be downloaded
        model.tokenizer.download_path = os.path.join(
            model.base_path, model.tokenizer.class_name)

        # Check if the tokenizer_path already exists
        if is_path_valid_for_download(model.tokenizer.download_path,
                                      overwrite):
            exit_error(
                f"Tokenizer '{model.tokenizer.download_path}' already exists.")

        # Processing options
        options = process_options(model.tokenizer.options or [])

        # Downloading the tokenizer
        tokenizer_downloaded = tokenizer_class_obj.from_pretrained(model.name,
                                                                   **options)
        tokenizer_downloaded.save_pretrained(model.tokenizer.download_path)

    except Exception as e:
        err = f"Error downloading tokenizer {model.tokenizer.class_name}: {e}"
        exit_error(err, ERROR_EXIT_TOKENIZER)


def is_path_valid_for_download(path: str, overwrite: bool) -> bool:
    """
    Check if the path is valid for downloading.

    Args:
        path (str): The path to check.
        overwrite (bool): Whether to overwrite existing files.

    Returns:
        bool: True if the path is valid for download, False otherwise.
    """
    return not overwrite and os.path.exists(path) and os.listdir(path)


def process_options(options_list: list) -> dict:
    """
    Process the options provided as a list of strings and convert them into a
        dictionary.

    Args:
        options_list (list): A list of options in the form of strings, where
            each string is in the format 'key=value'.

    Returns:
        dict: A dictionary containing the processed options, where keys are
            the option names and values are the corresponding values.
    """

    # Processed options
    options_dict = {}

    # Process every option
    for option in options_list:

        # Check if the option contains '='
        # Other options without affectation might exist but aren't handled
        # by this code until today
        if '=' not in option:
            exit_error(
                f"Invalid option format: {option}. "
                "Options must be in the form 'key=value'."
            )

        # Extract the dict item properties
        key, value = option.split('=')

        # Check if the value is a string in itself
        # (i.e. enclosed in single or double quotes)
        if (value.startswith("'") and value.endswith("'")) or \
                (value.startswith('"') and value.endswith('"')):
            # Remove the enclosing quotes and store as a string
            options_dict[key.strip()] = value[1:-1]
            continue

        try:
            # Attempt to evaluate the value
            evaluated_value = eval(value)

            # Evaluation succeed : saving the value
            options_dict[key.strip()] = evaluated_value

        # Evaluation failed because it could not find the requested
        # property as a value
        except NameError:

            try:
                # Attempt to import the value
                module_name, attr_name = value.rsplit('.', 1)
                module = importlib.import_module(module_name)

                # Attempt to evaluate the value
                evaluated_value = getattr(module, attr_name)

                # Evaluation succeed : saving the value
                options_dict[key.strip()] = evaluated_value

            # Missing import
            except ImportError:
                exit_error(
                    f"The required package for '{option}' wasn't found. "
                    "Please install the package."
                )

        # Evaluation failed
        except Exception as e:
            exit_error(f"Error evaluating value for key '{key.strip()}': {e}")

    return options_dict


def map_args_to_model(args) -> Model:
    """
    Maps command-line arguments to a Model object.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        Model: Model object representing the configuration.
    """

    # Mapping to tokenizer
    tokenizer = Tokenizer(args.tokenizer_class, args.tokenizer_options)

    # Mapping to model
    return Model(args.model_name, args.model_module, args.model_class,
                 args.model_options, args.access_token, tokenizer)


def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description="Script to download a specific model.")

    # Mandatory arguments related to the model
    parser.add_argument("models_path", type=str,
                        help="Path to the downloads directory")
    parser.add_argument("model_name", type=str,
                        help="Model name")
    parser.add_argument("model_module", type=str, help="Module name",
                        choices=AUTHORIZED_MODULE_NAMES)

    # Optional arguments regarding the model
    parser.add_argument("--access-token", type=str,
                        help="Access token for downloading the model")
    parser.add_argument("--model-class", type=str,
                        help="Class name within the module")
    parser.add_argument("--model-options", nargs="+", help="List of options")

    # Optional arguments regarding the model's tokenizer
    parser.add_argument("--tokenizer-class", type=str,
                        help="Tokenizer class name (only for transformers)")
    parser.add_argument(
        "--tokenizer-options", nargs="+",
        help="List of tokenizer options (only for transformers)"
    )

    # Global tags for the script
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing directories")
    parser.add_argument("--skip", type=str, help="Skip the download item",
                        choices=SKIP_ARGUMENTS)
    parser.add_argument("--emf-client", action="store_true",
                        help="If running from the emf-client")

    return parser.parse_args()


def main():
    """
    Main function to execute the download process based on the provided
        configuration file.
    """

    # Process arguments
    args = parse_arguments()

    # Map them into a model
    model = map_args_to_model(args)

    # Run download with specified arguments
    properties = model.download(args.models_path, args.skip, args.overwrite)

    # Running from emf-client:
    if args.emf_client:

        # Write model properties to stdout: the emf-client needs to get it
        # back to update the config file
        print(properties)


if __name__ == "__main__":
    main()  # pragma: no cover
