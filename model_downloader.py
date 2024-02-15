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

# Set up logging
logging.basicConfig(level=logging.INFO)


def exit_error(message, error=1):
    """
    Print an error message to the standard error stream and exit the script with a specified error code.

    Args:
        message (str): The error message to be printed.
        error (int, optional): The error code to exit the script with. Defaults to 1.

    Example:
        >>> exit_error("An error occurred.", 2)
        An error occurred.
        Script exits with status code 2.
    """
    print(message, file=sys.stderr)
    sys.exit(error)


def download(downloads_path, model_name, module_name, class_name, overwrite, options):
    """
    Downloads the model given data.

    Args:
        downloads_path (str): Path to the download's directory.
        model_name (str): The name of the model to download.
        module_name (str): The name of the module to use for downloading.
        class_name (str): The class within the module to use for downloading.
        overwrite (bool): Whether to overwrite the downloaded model if it exists.
        options (list): List of options used for downloading the model.

    Returns:
        None
    """

    # Check if the model name is not empty
    if model_name is None or model_name.strip() == '':
        exit_error(f"ERROR: Model '{model_name}' is invalid.")

    # Local path where the model will be downloaded
    model_path = os.path.join(downloads_path, model_name)

    # Check if the model_path already exists
    if not overwrite and os.path.exists(model_path):
        exit_error(f"Directory '{model_path}' already exists.")

    # Check if the module is authorized
    if module_name not in AUTHORIZED_MODULE_NAMES:
        exit_error(f"Module '{module_name}' is not authorized.")

    # Class is not provided, trying the default one
    if class_name is None or class_name.strip() == '':
        logging.warning("Module class not provided, using default but might fail.")
        class_name = model_config_default_class_for_module.get(module_name)

    # Processing options
    options_dict = process_options(options)

    try:
        # Transforming from strings to actual objects
        module_obj = globals()[module_name]
        class_obj = getattr(module_obj, class_name)

        # Downloading the model
        model = class_obj.from_pretrained(model_name, **options_dict)
        model.save_pretrained(model_path)

        # TODO : Tokenizer?
        if module_name == TRANSFORMERS:
            tokenizer_path = os.path.join(model_path, 'tokenizer')
            tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
            tokenizer.save_pretrained(tokenizer_path)

    except Exception as e:
        exit_error(f"Error while downloading model {model_name}: {e}", 2)


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


def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """

    parser = argparse.ArgumentParser(description="Script to download a specific model.")

    parser.add_argument("downloads_path", type=str, help="Path to the downloads directory")
    parser.add_argument("model_name", type=str, help="Model name")
    parser.add_argument("module_name", type=str, help="Module name")
    parser.add_argument("--class", dest="class_name", type=str, help="Class name (optional)")
    parser.add_argument("--overwrite", dest="overwrite", action="store_true", help="Overwrite existing directories")
    parser.add_argument("--options", nargs="+", help="List of options")

    return parser.parse_args()


def main():
    """
    Main function to execute the download process based on the provided configuration file.
    """

    # Process arguments
    args = parse_arguments()

    # Run download with specified arguments
    download(args.downloads_path, args.model_name, args.module_name, args.class_name, args.overwrite, args.options)


if __name__ == "__main__":
    main()
