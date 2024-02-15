import argparse
import logging
import os
import sys
import importlib

import diffusers
import transformers

DIFFUSERS = 'diffusers'
TRANSFORMERS = 'transformers'

# Authorized library names for download
AUTHORIZED_MODULE_NAMES = {DIFFUSERS, TRANSFORMERS}

# Map module name to a default class name
model_config_default_class_for_module = {
    DIFFUSERS: "DiffusionPipeline",
    TRANSFORMERS: "AutoModel",
}

# Transformers default tokenizer
TRANSFORMERS_DEFAULT_TOKENIZER_NAME = "AutoTokenizer"

# Set up logging
logging.basicConfig(level=logging.INFO)


def exit_error(message, error=1):
    """
    Print an error message to the standard error stream and exit the script with a specified error code.

    Args:
        message (str): The error message to be printed.
        error (int, optional): The error code to exit the script with. Defaults to 1.
    """
    print(message, file=sys.stderr)
    sys.exit(error)


class Tokenizer:
    """
    A class representing a Tokenizer.

    Attributes:
        name (str): The name of the tokenizer.
        options (list): List of options for the tokenizer.
    """

    def __init__(self, name=None, options=None):
        self.name = name or TRANSFORMERS_DEFAULT_TOKENIZER_NAME
        self.options = options or []


class Model:
    """
    A class representing a Model.

    Attributes:
        path (str): Path to the directory where the model will be downloaded.
        name (str): The name of the model to download.
        module (str): The name of the module to use for downloading the model.
        module_class (str, optional): The class within the module to use for downloading the model. Defaults to None.
        options (list, optional): List of options used for downloading the model. Defaults to None.
        tokenizer (Tokenizer, optional): Tokenizer object for the model. Defaults to None.
    """

    def __init__(self, path, name, module, module_class=None, options=None, tokenizer=None):
        self.path = path
        self.name = name
        self.module = module
        self.module_class = module_class
        self.options = options or []
        self.tokenizer = tokenizer

    def validate(self):
        """
        Validate the model.
        """

        # Check if the model name is not empty
        if not self.name.strip():
            exit_error(f"Model '{self.name}' is invalid.")

        # Check if the module is authorized
        if self.module not in AUTHORIZED_MODULE_NAMES:
            exit_error(f"Module '{self.module}' is not authorized. Must be one of {AUTHORIZED_MODULE_NAMES}")

    def download(self, overwrite=False):
        """
        Download the model.
        """

        # Validate mandatory arguments
        self.validate()

        # Local path where the model will be downloaded
        self.path = os.path.join(self.path, self.name)

        # Download the model
        download_model(self, overwrite)

        # Checking for tokenizer
        if self.module == TRANSFORMERS and self.tokenizer:
            # Download a tokenizer for the model
            download_tokenizer(self)


def download_model(model, overwrite=False):
    """
    Download the model.

    Args:
        model (Model): Model to be downloaded.
        overwrite (bool): Whether to overwrite the downloaded model if it exists.
    """

    # Check if the model_path already exists
    if not overwrite and os.path.exists(model.path):
        exit_error(f"Directory '{model.path}' already exists.")

    # Model class is not provided, trying the default one
    if model.module_class is None or model.module_class.strip() == '':
        logging.warning("Module class not provided, using default but might fail.")
        model.module_class = model_config_default_class_for_module.get(model.module)

    # Processing options
    options = process_options(model.options or [])

    try:
        # Transforming from strings to actual objects
        module_obj = globals()[model.module]
        model_class_obj = getattr(module_obj, model.module_class)

        # Downloading the model
        model_downloaded = model_class_obj.from_pretrained(model.name, **options)
        model_downloaded.save_pretrained(model.path)

    except Exception as e:
        exit_error(f"Error while downloading model {model.name}: {e}", 2)


def download_tokenizer(model):
    """
    Download a tokenizer for the model.

    Args:
        model (Model): Model to be downloaded.
    """

    try:

        # Retrieving tokenizer class from module
        tokenizer_class = getattr(transformers, model.tokenizer.name)

        # Local path where the tokenizer will be downloaded
        tokenizer_path = os.path.join(model.path, model.tokenizer.name)

        # TODO : check existence
        # TODO : if --tokenizer then check that model exists

        # Processing options
        options = process_options(model.tokenizer.options or [])

        # Downloading the tokenizer
        tokenizer_downloaded = tokenizer_class.from_pretrained(model.name, **options)
        tokenizer_downloaded.save_pretrained(tokenizer_path)

    except Exception as e:
        exit_error(f"Error while downloading tokenizer {model.tokenizer.name}: {e}", 3)


def process_options(options_list):
    """
    Process the options provided as a list of strings and convert them into a dictionary.

    Args:
        options_list (list): A list of options in the form of strings, where each string is in the format 'key=value'.

    Returns:
        dict: A dictionary containing the processed options, where keys are the option names and values are the corresponding values.
    """

    # Processed options
    options_dict = {}

    # Process every option
    for option in options_list:

        # Check if the option contains '='
        # Other options without affectation might exist but aren't handled by this code until today
        if '=' not in option:
            exit_error(f"Invalid option format: {option}. Options must be in the form 'key=value'.")

        # Extract the dict item properties
        key, value = option.split('=')

        # Check if the value is a string in itself (i.e. enclosed in single or double quotes)
        if (value.startswith("'") and value.endswith("'")) or (value.startswith('"') and value.endswith('"')):
            # Remove the enclosing quotes and store as a string
            options_dict[key.strip()] = value[1:-1]
            continue

        try:
            # Attempt to evaluate the value
            evaluated_value = eval(value)

            # Evaluation succeed : saving the value
            options_dict[key.strip()] = evaluated_value

        # Evaluation failed because it could not find the requested property as a value
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
                exit_error(f"The required package for '{option}' wasn't found. Please install the package.")

        # Evaluation failed
        except Exception as e:
            exit_error(f"Error evaluating value for key '{key.strip()}': {e}")

    return options_dict


def map_args_to_model(args):
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
    return Model(args.path, args.model_name, args.model_module, args.model_class, args.model_options, tokenizer)


def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """

    parser = argparse.ArgumentParser(description="Script to download a specific model.")

    # TODO : model part, tokenizer part and combination ?
    # TODO : --tokenizer tag to only add a tokenizer ?
    parser.add_argument("path", type=str, help="Path to the downloads directory")
    parser.add_argument("model_name", type=str, help="Model name")
    parser.add_argument("model_module", type=str, help=f"Module name", choices=AUTHORIZED_MODULE_NAMES)
    parser.add_argument("--model-class", type=str, help="Class name within the module")
    parser.add_argument("--model-options", nargs="+", help="List of options")
    parser.add_argument("--tokenizer-class", type=str, help="Tokenizer class name (only for transformers)")
    parser.add_argument("--tokenizer-options", nargs="+", help="List of tokenizer options (only for transformers)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing directories")

    return parser.parse_args()


def main():
    """
    Main function to execute the download process based on the provided configuration file.
    """

    # Process arguments
    args = parse_arguments()

    # Map them into a model
    model = map_args_to_model(args)

    # Run download with specified arguments
    model.download(args.overwrite)


if __name__ == "__main__":
    main()
