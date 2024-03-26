import argparse
import importlib
import json
import os
import sys

import diffusers
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

# Access token
KEY_ACCESS_TOKEN = "token"

# Error exit codes
ERROR_EXIT_DEFAULT = 1
ERROR_EXIT_MODEL = 2
ERROR_EXIT_MODEL_IMPORTS = 3
ERROR_EXIT_TOKENIZER = 4
ERROR_EXIT_TOKENIZER_IMPORTS = 5


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
        self.class_name = class_name
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

    def belongs_to_module(self, module: str) -> bool:
        """
        Check if the model belongs to a given module.

        Returns:
            bool: True if the model belongs to the module, False otherwise.
        """
        return self.module == module

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
        if self.belongs_to_module(TRANSFORMERS):
            self.download_path = os.path.join(
                self.base_path, TRANSFORMERS_DEFAULT_MODEL_DIRECTORY)

            # Local path where the tokenizer will be downloaded
            self.tokenizer.download_path = os.path.join(
                self.base_path, self.tokenizer.class_name)

    def process(self, models_path: str, skip: str = "",
                only_configuration: bool = False,
                overwrite: bool = False) -> str:
        """
        Process the model.

        Args:
            models_path (str): The base path where all the models are located.
            skip (str): Optional. Skips the download process of either the
                model or the tokenizer.
            only_configuration (bool): Optional. Whether to only get the
                configuration properties without downloading anything or not.
            overwrite (bool): Optional. Whether to overwrite the downloaded
                model if it exists.

        Returns:
            Program exits with error if the process fails.
            If it succeeds, it returns the JSON props used for downloading
            the model.
        """

        # Validate mandatory arguments
        self.validate()

        # Processing options
        options = process_options(self.options)

        # Processing access token
        access_token = process_access_token(options, self)

        # Set model class names
        set_class_names(self, access_token)

        # Build paths
        self.build_paths(models_path)

        # Output result
        result_dict = {"module": self.module}

        # Options
        options_tokenizer = {}

        if skip != DOWNLOAD_MODEL:
            # Adding downloaded model properties to result
            result_dict["class"] = self.class_name
            result_dict["path"] = self.download_path
            result_dict["options"] = get_options_for_json(options)

        if self.belongs_to_module(TRANSFORMERS) and skip != DOWNLOAD_TOKENIZER:
            # Processing options
            options_tokenizer = process_options(self.tokenizer.options)

            # Adding downloaded tokenizer properties to result
            result_dict["tokenizer"] = {
                "class": self.tokenizer.class_name,
                "path": self.tokenizer.download_path,
                "options": get_options_for_json(options_tokenizer)
            }

        # Execute download if requested
        if not only_configuration:
            self.download(skip, overwrite, options, options_tokenizer,
                          access_token)

        # Convert the dictionary to JSON
        return json.dumps(result_dict, indent=4)

    def download(self, skip: str, overwrite: bool,
                 options: dict, options_tokenizer: dict,
                 access_token: str | None) -> None:
        """
        Download the model.

        Args:
            skip (str): Skips the download process of either the model
                or the tokenizer.
            overwrite (bool): Whether to overwrite the downloaded model
                if it exists.
            options (dict): The options dictionary for the model.
            options_tokenizer (dict): The options dictionary for the tokenizer.
            access_token (str): The access token for the model
        """
        # Checking for model download
        if skip != DOWNLOAD_MODEL:
            # Downloading the model
            download_model(self, overwrite, options,
                           access_token)

        # Checking for tokenizer download
        if self.belongs_to_module(TRANSFORMERS) and skip != DOWNLOAD_TOKENIZER:
            # Download a tokenizer for the model
            download_transformers_tokenizer(
                self, overwrite, options_tokenizer)


def set_class_names(model: Model, access_token: str | None) -> None:
    """
    Set the appropriate model class name based on the model's module.
    And Set the appropriate tokenizer class name if needed.

    Args:
        model (Model): The model object.
        access_token (str): The access token for the model
    """
    if model.belongs_to_module(TRANSFORMERS):
        set_transformers_class_names(model, access_token)
    elif model.belongs_to_module(DIFFUSERS):
        set_diffusers_class_names(model, access_token)


def set_transformers_class_names(model: Model,
                                 access_token: str | None) -> None:
    """
    Set the appropriate model class for a Transformers module model
        and tokenizer.

    Args:
        model (Model): The model object.
        access_token (str): The access token for the model
    """
    try:
        # Get the configuration
        config = transformers.AutoConfig.from_pretrained(
            model.name, token=access_token)

        # Set model class name if not already set
        model.class_name = model.class_name or config.architectures[0] \
            if config.architectures and config.architectures[0] else (
            model_config_default_class_for_module[TRANSFORMERS]
        )

        # Set tokenizer class name if not already set
        # and config.tokenizer_class exists
        model.tokenizer.class_name = model.tokenizer.class_name or (
            config.tokenizer_class if config.tokenizer_class else
            TRANSFORMERS_DEFAULT_TOKENIZER_CLASS
        )
    except Exception:
        # Set default model class name if not already set
        model.class_name = model.class_name or \
                           model_config_default_class_for_module[TRANSFORMERS]

        # Set default tokenizer class name if not already set
        model.tokenizer.class_name = (model.tokenizer.class_name or
                                      TRANSFORMERS_DEFAULT_TOKENIZER_CLASS)


def set_diffusers_class_names(model: Model, access_token: str | None) -> None:
    """
    Set the appropriate model class for a Diffusers module model.

    Args:
        model (Model): The model object.
        access_token (str): The access token for the model
    """
    if model.class_name is not None and model.class_name != "":
        return

    try:
        # Get the configuration
        config = diffusers.DiffusionPipeline.load_config(
            model.name, use_auth_token=access_token)

        # get model class name from the configuration
        model.class_name = config['_class_name']
    except Exception:
        model.class_name = model_config_default_class_for_module[DIFFUSERS]


def download_model(model: Model, overwrite: bool, options: dict,
                   access_token: str | None) -> None:
    """
    Download the model.

    Args:
        model (Model): Model to be downloaded.
        overwrite (bool): Whether to overwrite the downloaded model if
            it exists.
        options: A dictionary containing options used for model downloading.
        access_token (str): The access token for the model
    """

    # Check if the model already exists at path
    if not is_path_valid_for_download(model.download_path, overwrite):
        exit_error(f"Model '{model.download_path}' already exists.")

    # Model class is not provided, trying the default one
    if model.class_name is None or model.class_name.strip() == '':
        model.class_name = model_config_default_class_for_module.get(
            model.module)

    # Transforming from strings to actual objects
    model_class_obj = None
    try:
        if model.module in sys.modules:
            module_obj = globals()[model.module]
        else:
            module_obj = importlib.import_module(model.module)
        model_class_obj = getattr(module_obj, model.class_name)
    except Exception as e:
        err = f"Error importing modules for model {model.name}: {e}"
        exit_error(err, ERROR_EXIT_MODEL_IMPORTS)

    # Downloading the model
    try:
        model_downloaded = model_class_obj.from_pretrained(
            model.name, token=access_token, **options)
        model_downloaded.save_pretrained(model.download_path)
    except Exception as e:
        err = f"Error downloading model {model.name}: {e}"
        exit_error(err, ERROR_EXIT_MODEL)


def download_transformers_tokenizer(model: Model, overwrite: bool,
                                    options: dict) -> None:
    """
    Download a transformers tokenizer for the model.

    Args:
        model (Model): Model to be downloaded.
        overwrite (bool): Whether to overwrite the downloaded model if
            it exists.
        options: A dictionary containing options used for model downloading.
    """

    # Retrieving tokenizer class from module
    tokenizer_class_obj = None
    try:
        # Model class is not provided, trying the default one
        if (model.tokenizer.class_name is None or
                model.tokenizer.class_name.strip() == ''):
            model.tokenizer.class_name = TRANSFORMERS_DEFAULT_TOKENIZER_CLASS

        # Retrieving tokenizer class from module
        tokenizer_class_obj = getattr(transformers, model.tokenizer.class_name)
    except Exception as e:
        err = f"Error importing tokenizer {model.tokenizer.class_name}: {e}"
        exit_error(err, ERROR_EXIT_TOKENIZER_IMPORTS)

    # Check if the tokenizer_path already exists
    if not is_path_valid_for_download(
            model.tokenizer.download_path, overwrite):
        err = f"Tokenizer '{model.tokenizer.download_path}' already exists."
        exit_error(err)

    # Downloading the tokenizer
    try:
        tokenizer_downloaded = tokenizer_class_obj.from_pretrained(
            model.name, **options)
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
    return overwrite or not os.path.exists(path) or not os.listdir(path)


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


def process_access_token(options: dict, model: Model) -> str | None:
    """
    Process the access token since it can be provided through options and flags

    Args:
        options (dict): A dictionary containing the processed options.
        model (Model): Model to be downloaded.

    Returns:
        str: The value of the access token (if provided).
    """

    # If conflicting access tokens are provided, raise an error
    options_access_token = options.get(KEY_ACCESS_TOKEN)
    if (options_access_token and model.access_token and
            options_access_token == model.access_token):
        exit_error("Conflicting access tokens provided. "
                   "Please provide only one access token.")

    access_token = ""

    # Access token provided through options
    if options_access_token:
        options.pop(KEY_ACCESS_TOKEN)
        access_token = options_access_token

    # Access token provided through flags
    elif model.access_token:
        access_token = model.access_token

    return access_token


def get_options_for_json(options_dict: dict) -> dict:
    """
    Prepares a dictionary containing options for conversion to JSON.
    Args:
      options_dict (dict): A dictionary containing options as key-value pairs.
    Returns:
      dict: A new dictionary with the same keys but with values prepared for
            JSON serialization (strings with quotes for string values).
    """

    # Create a shallow copy of the input dictionary
    options = options_dict.copy()

    for key, value in options.items():
        if isinstance(value, str):
            options[key] = "\"{}\"".format(value)
        else:
            options[key] = str(value)
    return options


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
    parser.add_argument("model_module", type=str,
                        help="Module name", choices=AUTHORIZED_MODULE_NAMES)

    # Optional arguments regarding the model
    parser.add_argument("--model-class", type=str,
                        help="Class name within the module")
    parser.add_argument("--model-options", nargs="+",
                        help="List of options")
    parser.add_argument("--access-token", type=str,
                        help="Access token for downloading the model")

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
    parser.add_argument("--skip", type=str,
                        help="Skip the download item", choices=SKIP_ARGUMENTS)
    parser.add_argument("--emf-client", action="store_true",
                        help="If running from the emf-client")
    parser.add_argument("--only-configuration",
                        action="store_true",
                        help="Get model configuration only")

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

    # Run process with specified arguments
    properties = model.process(args.models_path, args.skip,
                               args.only_configuration, args.overwrite)

    # Running from emf-client:
    if args.emf_client:
        # Write model properties to stdout: the emf-client needs to get it
        # back to update the config file
        print(properties)


if __name__ == "__main__":
    main()  # pragma: no cover
