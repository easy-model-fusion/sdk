import argparse
import json
import os
import sys
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..")))

from downloader import (  # noqa: E402
    Model,
    Tokenizer,
    download_model,
    set_transformers_class_names,
    set_class_names,
    set_diffusers_class_names,
    download_transformers_tokenizer,
    is_path_valid_for_download,
    process_options,
    process_access_token,
    get_options_for_json,
    map_args_to_model,
    main,
    exit_error,
    DIFFUSERS,
    TRANSFORMERS,
    TRANSFORMERS_DEFAULT_MODEL_DIRECTORY,
    DOWNLOAD_MODEL,
    DOWNLOAD_TOKENIZER,
    KEY_ACCESS_TOKEN,
    ERROR_EXIT_DEFAULT,
    ERROR_EXIT_MODEL,
    ERROR_EXIT_MODEL_IMPORTS,
    ERROR_EXIT_TOKENIZER,
    ERROR_EXIT_TOKENIZER_IMPORTS
)


class TestDownloader(unittest.TestCase):

    @patch('builtins.print')
    @patch('sys.exit')
    def test_exit_error(self, mock_exit, mock_print):
        # Init
        message = "Test message"
        code = 42

        # Execute
        exit_error(message, code)

        # Assert
        mock_print.assert_called_once_with(message, file=sys.stderr)
        mock_exit.assert_called_once_with(code)

    @patch('builtins.print')
    @patch('sys.exit')
    def test_exit_error_default_code(self, mock_exit, mock_print):
        # Init
        message = "Test message"

        # Execute
        exit_error(message)

        # Assert
        mock_print.assert_called_once_with(message, file=sys.stderr)
        mock_exit.assert_called_once_with(ERROR_EXIT_DEFAULT)

    @patch('builtins.print')
    @patch('sys.exit')
    def test_exit_error_default_code_empty(self, mock_exit, mock_print):
        # Init
        message = ""

        # Execute
        exit_error(message)

        # Assert
        mock_print.assert_called_once_with(message, file=sys.stderr)
        mock_exit.assert_called_once_with(ERROR_EXIT_DEFAULT)

    def test_model_validation_valid(self):
        # Init
        valid_model = Model(name="valid_model", module="some_module")

        # Execute : no errors = success
        valid_model.validate()

    def test_model_validation_invalid(self):
        # Init
        invalid_model = Model(name="", module="some_module")

        # Execute : error = success
        with self.assertRaises(SystemExit) as context:
            invalid_model.validate()
        self.assertEqual(context.exception.code, 1)

    def test_belongs_to_module_true(self):
        # Init
        transformers_model = Model(name="", module=TRANSFORMERS)

        # Execute & Assert
        self.assertTrue(transformers_model.belongs_to_module(TRANSFORMERS))

    def test_belongs_to_module_false(self):
        # Init
        non_transformers_model = Model(name="", module=DIFFUSERS)

        # Execute & Assert
        self.assertFalse(non_transformers_model.
                         belongs_to_module(TRANSFORMERS))

    def test_build_paths_default(self):
        # Init
        model_name = "TestModel"
        models_path = "path/to/models"
        model = Model(name=model_name, module="")

        # Prepare
        expected_path = os.path.join(models_path, model_name)

        # Execute
        model.build_paths(models_path)

        # Assert
        self.assertEqual(model.base_path, expected_path)
        self.assertEqual(model.download_path, expected_path)

    def test_build_paths_transformers(self):
        # Init
        model_name = "TestModel"
        models_path = "path/to/models"
        tokenizer = Tokenizer(class_name="TransformersTokenizer")
        model = Model(
            name=model_name, module=TRANSFORMERS, tokenizer=tokenizer)

        # Prepare
        expected_base_path = os.path.join(models_path, model_name)
        expected_download_path = os.path.join(
            expected_base_path, TRANSFORMERS_DEFAULT_MODEL_DIRECTORY)

        # Execute
        model.build_paths(models_path)

        # Assert
        self.assertEqual(model.base_path, expected_base_path)
        self.assertEqual(model.download_path, expected_download_path)

    @patch('os.path.exists')
    @patch('os.listdir')
    def test_is_path_valid_for_download_valid_overwrite(
            self, mock_listdir, mock_exists):
        # Init
        models_path = "path/to/models"
        overwrite = True

        # Execute
        valid = is_path_valid_for_download(models_path, overwrite)

        # Assert
        self.assertTrue(valid)
        mock_exists.assert_not_called()
        mock_listdir.assert_not_called()

    @patch('os.path.exists', return_value=False)
    @patch('os.listdir')
    def test_is_path_valid_for_download_valid_not_exist(
            self, mock_listdir, mock_exists):
        # Init
        models_path = "path/to/models"
        overwrite = False

        # Execute
        valid = is_path_valid_for_download(models_path, overwrite)

        # Assert
        self.assertTrue(valid)
        mock_exists.assert_called_once()
        mock_listdir.assert_not_called()

    @patch('os.path.exists', return_value=True)
    @patch('os.listdir', return_value=[])
    def test_is_path_valid_for_download_valid_empty(
            self, mock_listdir, mock_exists):
        # Init
        models_path = "path/to/models"
        overwrite = False

        # Execute
        valid = is_path_valid_for_download(models_path, overwrite)

        # Assert
        self.assertTrue(valid)
        mock_exists.assert_called_once()
        mock_listdir.assert_called_once()

    @patch('os.path.exists', return_value=True)
    @patch('os.listdir', return_value=['file 1'])
    def test_is_path_valid_for_download_invalid(
            self, mock_listdir, mock_exists):
        # Init
        models_path = "path/to/models"
        overwrite = False

        # Execute
        valid = is_path_valid_for_download(models_path, overwrite)

        # Assert
        self.assertFalse(valid)
        mock_exists.assert_called_once()
        mock_listdir.assert_called_once()

    def test_process_options(self):
        # Init
        options_list = ["key1='value1'", "key2='value2'", "key3=3"]

        # Execute
        options_dict = process_options(options_list)

        # Assert
        self.assertEqual(
            options_dict, {"key1": "value1", "key2": "value2", "key3": 3})

    def test_process_options_invalid_format(self):
        # Init
        options_list = ["invalid_option"]

        # Execute : error = success
        with self.assertRaises(SystemExit) as context:
            process_options(options_list)
        self.assertEqual(context.exception.code, ERROR_EXIT_DEFAULT)

    @patch('builtins.eval', MagicMock(return_value="evaluated_value"))
    def test_process_options_evaluated_value(self):
        # Init
        options_list = ["key=expression"]

        # Execute
        options_dict = process_options(options_list)

        # Assert
        self.assertEqual(options_dict, {"key": "evaluated_value"})

    @patch('builtins.eval', side_effect=Exception("Evaluation failed"))
    def test_process_options_evaluated_value_error(self, mock_eval):
        # Init
        options_list = ["key=expression"]

        # Execute : error = success
        with self.assertRaises(SystemExit) as context:
            process_options(options_list)
        self.assertEqual(context.exception.code, ERROR_EXIT_DEFAULT)

    @patch('importlib.import_module', MagicMock(return_value=MagicMock()))
    def test_process_options_import_module(self):
        # Init
        options_list = ["key=module.attribute"]

        # Execute
        options_dict = process_options(options_list)

        # Assert
        self.assertIn("key", options_dict)
        self.assertIsInstance(options_dict["key"], MagicMock)

    @patch('importlib.import_module', side_effect=ImportError("Import failed"))
    def test_process_options_import_module_error(self, mock_import_module):
        # Init
        options_list = ["key=module.attribute"]

        # Execute : error = success
        with self.assertRaises(SystemExit) as context:
            process_options(options_list)
        self.assertEqual(context.exception.code, 1)

    def test_process_access_token_error(self):
        # Init
        access_token = "token"
        options = {KEY_ACCESS_TOKEN: access_token}
        model = Model(name="TestModel", module="")
        model.access_token = access_token

        # Execute : error = success
        with self.assertRaises(SystemExit) as context:
            process_access_token(options, model)
        self.assertEqual(context.exception.code, ERROR_EXIT_DEFAULT)

    def test_process_access_token_from_options(self):
        # Init
        access_token = "token"
        options = {KEY_ACCESS_TOKEN: access_token}
        model = Model(name="TestModel", module="")

        # Execute
        result = process_access_token(options, model)

        # Assert
        self.assertEqual(access_token, result)

    def test_process_access_token_from_model(self):
        # Init
        access_token = "token"
        model = Model(name="TestModel", module="")
        model.access_token = access_token

        # Execute
        result = process_access_token({}, model)

        # Assert
        self.assertEqual(access_token, result)

    def test_process_access_token_missing(self):
        # Init
        model = Model(name="TestModel", module="")

        # Execute
        result = process_access_token({}, model)

        # Assert
        self.assertEqual("", result)

    def test_get_options_for_json_input_empty(self):
        # Init
        input_options = {}
        expected_options = {}

        # Execute
        output_options = get_options_for_json(input_options)

        # Assert
        self.assertEqual(expected_options, output_options)

    def test_get_options_for_json_value_string(self):
        # Init
        input_options = {
            "key": "value1"
        }
        expected_options = {
            "key": "\"value1\""
        }

        # Execute
        output_options = get_options_for_json(input_options)

        # Assert
        self.assertEqual(expected_options, output_options)

    def test_get_options_for_json_value_not_string(self):
        # Init
        input_options = {
            "key": True
        }
        expected_options = {
            "key": "True"
        }

        # Execute
        output_options = get_options_for_json(input_options)

        # Assert
        self.assertEqual(expected_options, output_options)

    @patch('downloader.is_path_valid_for_download', return_value=False)
    @patch('downloader.process_access_token')
    @patch('transformers.models.auto.modeling_auto.AutoModel.from_pretrained')
    def test_download_model_with_path_invalid(
            self, mock_from_pretrained, mock_process_access_token,
            mock_is_path_valid_for_download):
        # Init
        model = Model(name="TestModel", module="")
        options = {"key1": "value1"}

        # Execute : error = success
        with self.assertRaises(SystemExit) as context:
            download_model(model, overwrite=False, options=options,
                           access_token=None)
        self.assertEqual(context.exception.code, ERROR_EXIT_DEFAULT)

        # Assert
        mock_is_path_valid_for_download.assert_called_once()
        mock_process_access_token.assert_not_called()
        mock_from_pretrained.assert_not_called()

    @patch('downloader.is_path_valid_for_download', return_value=True)
    @patch('transformers.models.auto.modeling_auto.AutoModel.from_pretrained')
    def test_download_model_with_objects_error(
            self, mock_from_pretrained,
            mock_is_path_valid_for_download):
        # Init
        model = Model(name="TestModel", module="")
        options = {"key1": "value1"}

        # Execute : error = success
        with self.assertRaises(SystemExit) as context:
            download_model(model, overwrite=False, options=options,
                           access_token=None)
        self.assertEqual(context.exception.code, ERROR_EXIT_MODEL_IMPORTS)

        # Assert
        mock_is_path_valid_for_download.assert_called_once()
        mock_from_pretrained.assert_not_called()

    @patch('downloader.is_path_valid_for_download', return_value=True)
    @patch('transformers.models.auto.modeling_auto.AutoModel.from_pretrained'
           '', side_effect=Exception("Download failed"))
    def test_download_model_with_from_pretrained_error(
            self, mock_from_pretrained,
            mock_is_path_valid_for_download):
        # Init
        model = Model(name="TestModel", module=TRANSFORMERS)
        options = {"key1": "value1"}

        # Execute : error = success
        with self.assertRaises(SystemExit) as context:
            download_model(model, overwrite=False, options=options,
                           access_token=None)
        self.assertEqual(context.exception.code, ERROR_EXIT_MODEL)

        # Assert
        mock_is_path_valid_for_download.assert_called_once()
        mock_from_pretrained.assert_called_once()

    @patch('downloader.is_path_valid_for_download', return_value=True)
    @patch('transformers.models.auto.modeling_auto.AutoModel.from_pretrained')
    def test_download_model_with_save_pretrained_error(
            self, mock_from_pretrained,
            mock_is_path_valid_for_download):
        # Mockers : save_pretrained
        mock_save_pretrained = MagicMock(side_effect=Exception("Save failed"))

        # Adding save_pretrained to from_pretrained returned value
        data_from_pretrained = MagicMock()
        data_from_pretrained.save_pretrained = mock_save_pretrained
        mock_from_pretrained.return_value = data_from_pretrained

        # Init
        model = Model(name="TestModel", module=TRANSFORMERS)
        options = {"key1": "value1"}

        # Execute : error = success
        with self.assertRaises(SystemExit) as context:
            download_model(model, overwrite=False, options=options,
                           access_token=None)
        self.assertEqual(context.exception.code, ERROR_EXIT_MODEL)

        # Assert
        mock_is_path_valid_for_download.assert_called_once()
        mock_from_pretrained.assert_called_once()
        mock_save_pretrained.assert_called_once()

    @patch('downloader.is_path_valid_for_download', return_value=True)
    @patch('transformers.models.auto.modeling_auto.AutoModel.from_pretrained')
    def test_download_model_success(
            self, mock_from_pretrained,
            mock_is_path_valid_for_download):
        # Mockers : save_pretrained
        mock_save_pretrained = MagicMock(return_value=None)

        # Adding save_pretrained to from_pretrained returned value
        data_from_pretrained = MagicMock()
        data_from_pretrained.save_pretrained = mock_save_pretrained
        mock_from_pretrained.return_value = data_from_pretrained

        # Init
        model = Model(name="TestModel", module=TRANSFORMERS)
        model.options = ["key1='value1'"]
        options = {"key1": "value1"}

        # Execute :
        download_model(model, overwrite=False, options=options,
                       access_token=None)

        # Assert
        mock_is_path_valid_for_download.assert_called_once()
        mock_from_pretrained.assert_called_once()
        mock_save_pretrained.assert_called_once()

    @patch('downloader.is_path_valid_for_download', return_value=False)
    @patch('transformers.models.auto.tokenization_auto.AutoTokenizer'
           '.from_pretrained')
    def test_download_transformers_tokenizer_with_path_invalid(
            self, mock_from_pretrained,
            mock_is_path_valid_for_download):
        # Init
        model = Model(name="TestModel", module=TRANSFORMERS)
        model.base_path = "path/to/model"
        tokenizer = Tokenizer(class_name="AutoTokenizer")
        model.tokenizer = tokenizer
        options = {"key1": "value1"}

        # Execute : error = success
        with self.assertRaises(SystemExit) as context:
            download_transformers_tokenizer(
                model, overwrite=False, options=options)
        self.assertEqual(context.exception.code, ERROR_EXIT_DEFAULT)

        # Assert
        mock_is_path_valid_for_download.assert_called_once()
        mock_from_pretrained.assert_not_called()

    @patch('downloader.is_path_valid_for_download', return_value=True)
    @patch('transformers.models.auto.tokenization_auto.AutoTokenizer'
           '.from_pretrained')
    def test_download_transformers_tokenizer_with_objects_error(
            self, mock_from_pretrained,
            mock_is_path_valid_for_download):
        # Init
        model = Model(name="TestModel", module="")
        model.tokenizer = Tokenizer(class_name="error")
        options = {"key1": "value1"}

        # Execute : error = success
        with self.assertRaises(SystemExit) as context:
            download_transformers_tokenizer(
                model, overwrite=False, options=options)
        self.assertEqual(context.exception.code, ERROR_EXIT_TOKENIZER_IMPORTS)

        # Assert
        mock_is_path_valid_for_download.assert_not_called()
        mock_from_pretrained.assert_not_called()

    @patch('downloader.is_path_valid_for_download', return_value=True)
    @patch('transformers.models.auto.tokenization_auto.AutoTokenizer'
           '.from_pretrained', side_effect=Exception("Download failed"))
    def test_download_transformers_with_from_pretrained_error(
            self, mock_from_pretrained,
            mock_is_path_valid_for_download):
        # Init
        model = Model(name="TestModel", module=TRANSFORMERS)
        model.base_path = "path/to/model"
        tokenizer = Tokenizer(class_name="AutoTokenizer")
        model.tokenizer = tokenizer
        options = {"key1": "value1"}

        # Execute : error = success
        with self.assertRaises(SystemExit) as context:
            download_transformers_tokenizer(
                model, overwrite=False, options=options)
        self.assertEqual(context.exception.code, ERROR_EXIT_TOKENIZER)

        # Assert
        mock_is_path_valid_for_download.assert_called_once()
        mock_from_pretrained.assert_called_once()

    @patch('downloader.is_path_valid_for_download', return_value=True)
    @patch('transformers.models.auto.tokenization_auto.AutoTokenizer'
           '.from_pretrained')
    def test_download_transformers_with_save_pretrained_error(
            self, mock_from_pretrained,
            mock_is_path_valid_for_download):
        # Mockers : save_pretrained
        mock_save_pretrained = MagicMock(side_effect=Exception("Save failed"))

        # Adding save_pretrained to from_pretrained returned value
        data_from_pretrained = MagicMock()
        data_from_pretrained.save_pretrained = mock_save_pretrained
        mock_from_pretrained.return_value = data_from_pretrained

        # Init
        model = Model(name="TestModel", module=TRANSFORMERS)
        model.base_path = "path/to/model"
        tokenizer = Tokenizer(class_name="AutoTokenizer")
        model.tokenizer = tokenizer
        options = {"key1": "value1"}

        # Execute : error = success
        with self.assertRaises(SystemExit) as context:
            download_transformers_tokenizer(
                model, overwrite=False, options=options)
        self.assertEqual(context.exception.code, ERROR_EXIT_TOKENIZER)

        # Assert
        mock_is_path_valid_for_download.assert_called_once()
        mock_from_pretrained.assert_called_once()
        mock_save_pretrained.assert_called_once()

    @patch('downloader.is_path_valid_for_download', return_value=True)
    @patch('transformers.models.auto.tokenization_auto.AutoTokenizer'
           '.from_pretrained')
    def test_download_transformers_witouht_class_name(
            self, mock_from_pretrained,
            mock_is_path_valid_for_download):
        # Mockers : save_pretrained
        mock_save_pretrained = MagicMock(return_value=None)

        # Adding save_pretrained to from_pretrained returned value
        data_from_pretrained = MagicMock()
        data_from_pretrained.save_pretrained = mock_save_pretrained
        mock_from_pretrained.return_value = data_from_pretrained

        # Init
        model = Model(name="TestModel", module=TRANSFORMERS)
        model.base_path = "path/to/model"
        tokenizer = Tokenizer()
        options = {"key1": "value1"}
        tokenizer.options = options
        model.tokenizer = tokenizer

        # Execute :
        download_transformers_tokenizer(
            model, overwrite=False, options=options)

        # Assert
        self.assertEqual(model.tokenizer.class_name, 'AutoTokenizer')
        mock_is_path_valid_for_download.assert_called_once()
        mock_from_pretrained.assert_called_once()
        mock_save_pretrained.assert_called_once()

    @patch('downloader.is_path_valid_for_download', return_value=True)
    @patch('transformers.models.auto.tokenization_auto.AutoTokenizer'
           '.from_pretrained')
    def test_download_transformers_success(
            self, mock_from_pretrained,
            mock_is_path_valid_for_download):
        # Mockers : save_pretrained
        mock_save_pretrained = MagicMock(return_value=None)

        # Adding save_pretrained to from_pretrained returned value
        data_from_pretrained = MagicMock()
        data_from_pretrained.save_pretrained = mock_save_pretrained
        mock_from_pretrained.return_value = data_from_pretrained

        # Init
        model = Model(name="TestModel", module=TRANSFORMERS)
        model.base_path = "path/to/model"
        tokenizer = Tokenizer(class_name="AutoTokenizer")
        options = {"key1": "value1"}
        tokenizer.options = options
        model.tokenizer = tokenizer

        # Execute :
        download_transformers_tokenizer(
            model, overwrite=False, options=options)

        # Assert
        mock_is_path_valid_for_download.assert_called_once()
        mock_from_pretrained.assert_called_once()
        mock_save_pretrained.assert_called_once()

    @patch('downloader.download_model')
    def test_download_model_skip_success(self, mock_download_model):
        # Init
        model = Model(name="TestModel", module="module")
        model.download_path = "path/to/model"
        model.class_name = "class_name"
        model.options = ["test='test'"]
        model.validate = MagicMock()
        model.build_paths = MagicMock()

        # Execute
        model.download(overwrite=True, skip=DOWNLOAD_TOKENIZER,
                       options={"test": "\"test\""}, options_tokenizer={},
                       access_token=None)

        # Assert
        mock_download_model.assert_called_once()

    @patch('downloader.download_transformers_tokenizer')
    def test_download_tokenizer_skip_success(
            self, mock_download_transformers_tokenizer):
        # Init
        model = Model(name="TestModel", module=TRANSFORMERS)
        tokenizer = Tokenizer(class_name="AutoTokenizer")
        tokenizer.download_path = "path/to/tokenizer"
        tokenizer.options = ["test='test'"]
        model.tokenizer = tokenizer
        model.validate = MagicMock()
        model.build_paths = MagicMock()

        # Execute
        model.download(overwrite=True, skip=DOWNLOAD_MODEL,
                       options={"test": "\"test\""}, options_tokenizer={},
                       access_token=None)

        # Assert
        mock_download_transformers_tokenizer.assert_called_once()

    @patch('downloader.download_model', side_effect=SystemExit)
    def test_download_model_error(self, mock_download_model):
        # Init
        model = Model(name="TestModel", module=TRANSFORMERS)
        model.validate = MagicMock()
        model.build_paths = MagicMock()

        # Execute : error = success
        with self.assertRaises(SystemExit):
            model.download(overwrite=True, skip=DOWNLOAD_TOKENIZER,
                           options={}, options_tokenizer={"options": {}},
                           access_token=None)

        # Assert
        mock_download_model.assert_called_once()

    @patch('downloader.download_transformers_tokenizer',
           side_effect=SystemExit)
    def test_download_tokenizer_error(
            self, mock_download_transformers_tokenizer):
        # Init
        model = Model(name="TestModel", module=TRANSFORMERS)
        model.validate = MagicMock()
        model.build_paths = MagicMock()

        # Execute : error = success
        with self.assertRaises(SystemExit):
            model.download(overwrite=True, skip=DOWNLOAD_MODEL,
                           options={},
                           options_tokenizer={"tokenizer": {"options": {}}},
                           access_token=None)

        # Assert
        mock_download_transformers_tokenizer.assert_called_once()

    def test_map_args_to_model(self):
        # Init
        model_name = "test_model"
        model_module = "some_module"
        model_class = "TestClass"
        model_options = ["key1=value1"]
        access_token = "token"
        tokenizer_class = "TestTokenizer"
        tokenizer_options = ["key2=value2"]

        args = argparse.Namespace(
            model_name=model_name,
            model_module=model_module,
            model_class=model_class,
            model_options=model_options,
            access_token=access_token,
            tokenizer_class=tokenizer_class,
            tokenizer_options=tokenizer_options,
        )

        # Execute
        model = map_args_to_model(args)

        # Assert model
        self.assertEqual(model.name, model_name)
        self.assertEqual(model.module, model_module)
        self.assertEqual(model.class_name, model_class)
        self.assertEqual(model.options, model_options)
        self.assertEqual(model.access_token, access_token)

        # Assert tokenizer
        self.assertIsInstance(model.tokenizer, Tokenizer)
        self.assertEqual(model.tokenizer.class_name, tokenizer_class)
        self.assertEqual(model.tokenizer.options, tokenizer_options)

    @patch('downloader.process_options')
    @patch('downloader.Model.download', return_value=None)
    @patch('builtins.print')
    def test_main(self, mock_print, mock_model_download, mock_process_options):
        # Init
        args = argparse.Namespace(
            models_path="path/to/models",
            model_name="test_model",
            model_module="some_module",
            model_class="TestClass",
            model_options=["key=test"],
            access_token="token",
            tokenizer_class="TestTokenizer",
            tokenizer_options=["key=test"],
            overwrite=False,
            skip="tokenizer",
            emf_client=False,
            only_configuration=False,
        )

        # Options
        expected_options = {"key": "test"}
        mock_process_options.return_value = expected_options

        # Execute
        with patch('sys.argv', ['script_name']):
            with patch('argparse.ArgumentParser.parse_args',
                       return_value=args):
                main()

        # Assert
        mock_process_options.assert_called()
        mock_model_download.assert_called_once()
        mock_print.assert_not_called()

    @patch('downloader.Model.process')
    @patch('builtins.print')
    def test_main_emf_client(self, mock_print, mock_model_process):
        # Init
        args = argparse.Namespace(
            models_path="path/to/models",
            model_name="test_model",
            model_module="some_module",
            model_class="TestClass",
            model_options=["key1=value1"],
            access_token="token",
            tokenizer_class="TestTokenizer",
            tokenizer_options=["key2=value2"],
            overwrite=False,
            skip="tokenizer",
            emf_client=True,
            only_configuration=False
        )

        # Execute
        with patch('sys.argv', ['script_name']):
            with patch('argparse.ArgumentParser.parse_args',
                       return_value=args):
                main()

        # Assert
        mock_model_process.assert_called_once()
        mock_print.assert_called_once()

    @patch('diffusers.DiffusionPipeline.load_config',
           return_value={'_class_name': 'TestPipeline'})
    def test_set_diffusers_class_names(self, mock_load_config):
        # Init
        model = Model(name="TestModel", module=DIFFUSERS)

        # Execute
        set_diffusers_class_names(model, access_token=None)

        # Assert
        mock_load_config.assert_called_once()
        self.assertEqual(model.class_name, 'TestPipeline')

    @patch('diffusers.DiffusionPipeline.load_config',
           side_effect=Exception("Mocked exception from load config"))
    def test_set_diffusers_class_names_with_exception(self, mock_load_config):
        # Init
        model = Model(name="TestModel", module=DIFFUSERS)

        # Execute
        set_diffusers_class_names(model, access_token=None)

        # Assert
        mock_load_config.assert_called_once()
        self.assertEqual(model.class_name, 'DiffusionPipeline')

    @patch('diffusers.DiffusionPipeline.load_config',
           return_value={'_class_name': 'TestPipeline'})
    def test_set_diffusers_class_names_with_configured_model(self,
                                                             mock_load_config):
        # Init
        model = Model(name="TestModel", module=DIFFUSERS,
                      class_name="TestModel")

        # Execute
        set_diffusers_class_names(model, access_token=None)

        # Assert
        mock_load_config.assert_not_called()
        self.assertEqual(model.class_name, 'TestModel')

    @patch('transformers.AutoConfig.from_pretrained',
           return_value=MagicMock(architectures=['T5Model'],
                                  tokenizer_class='TokenizerClass'))
    def test_set_transformers_class_names(self, mock_load_config):
        # Init
        model = Model(name="TestModel", module=TRANSFORMERS,
                      tokenizer=Tokenizer())

        # Execute
        set_transformers_class_names(model, access_token=None)

        # Assert
        mock_load_config.assert_called_once()
        self.assertEqual(model.class_name, 'T5Model')
        self.assertEqual(model.tokenizer.class_name, 'TokenizerClass')

    @patch('transformers.AutoConfig.from_pretrained',
           return_value=MagicMock(architectures=['T5Model'],
                                  tokenizer_class=None))
    def test_set_transformers_class_names_with_default_tokenizer(
            self, mock_load_config
    ):
        # Init
        model = Model(name="TestModel", module=TRANSFORMERS,
                      tokenizer=Tokenizer())

        # Execute
        set_transformers_class_names(model, access_token=None)

        # Assert
        mock_load_config.assert_called_once()
        self.assertEqual(model.class_name, 'T5Model')
        self.assertEqual(model.tokenizer.class_name, 'AutoTokenizer')

    @patch('transformers.AutoConfig.from_pretrained',
           return_value=MagicMock(architectures=['T5Model'],
                                  tokenizer_class='TokenizerClass'))
    def test_set_transformers_class_names_with_configured_model(
            self, mock_load_config
    ):
        # Init
        model = Model(name="TestModel", module=TRANSFORMERS,
                      class_name='TestModel',
                      tokenizer=Tokenizer('TestTokenizer'))

        # Execute
        set_transformers_class_names(model, access_token=None)

        # Assert
        mock_load_config.assert_called_once()
        self.assertEqual(model.class_name, 'TestModel')
        self.assertEqual(model.tokenizer.class_name, 'TestTokenizer')

    @patch('transformers.AutoConfig.from_pretrained',
           side_effect=Exception("Mocked exception from load config"))
    def test_set_transformers_class_names_with_exception(
            self, mock_load_config
    ):
        # Init
        model = Model(name="TestModel", module=TRANSFORMERS,
                      tokenizer=Tokenizer('TestTokenizer'))

        # Execute
        set_transformers_class_names(model, access_token=None)

        # Assert
        mock_load_config.assert_called_once()
        self.assertEqual(model.class_name, 'AutoModel')
        self.assertEqual(model.tokenizer.class_name, 'TestTokenizer')

    @patch('downloader.set_transformers_class_names', return_value=None)
    @patch('downloader.set_diffusers_class_names', return_value=None)
    def test_set_class_names_for_transformers_model(
            self, mock_set_diffusers_class_names, mock_transformer_class_names
    ):
        # Init
        model = Model(name="TestModel", module=TRANSFORMERS)

        # Execute
        set_class_names(model, access_token=None)

        # Assert
        mock_transformer_class_names.assert_called_once()
        mock_set_diffusers_class_names.assert_not_called()

    @patch('downloader.set_transformers_class_names', return_value=None)
    @patch('downloader.set_diffusers_class_names', return_value=None)
    def test_set_class_names_for_diffusers_model(
            self, mock_set_diffusers_class_names, mock_transformer_class_names
    ):
        # Init
        model = Model(name="TestModel", module=DIFFUSERS)

        # Execute
        set_class_names(model, access_token=None)

        # Assert
        mock_set_diffusers_class_names.assert_called_once()
        mock_transformer_class_names.assert_not_called()

    @patch('downloader.process_options')
    @patch('downloader.process_access_token', return_value="")
    @patch('downloader.set_class_names', return_value=None)
    @patch('downloader.Model.download', return_value=None)
    @patch('downloader.Model.build_paths', return_value=None)
    @patch('downloader.Model.belongs_to_module', return_value=True)
    def test_process_with_only_configuration(
            self, mock_belongs_to_module, mock_build_paths,
            mock_download, mock_set_class_names, mock_process_access_token,
            mock_process_options
    ):
        # Options
        input_options = ["key='test'"]
        expected_options = {"key": "\"test\""}

        # init
        model = Model(name="TestModel", module=TRANSFORMERS,
                      class_name="TestClass", options=input_options)
        model.tokenizer = Tokenizer(class_name="TokenizerClass",
                                    options=input_options)

        # Options
        mock_process_options.return_value = expected_options

        # Prepare
        expected_result = {
            "module": model.module,
            "class": model.class_name,
            "path": model.download_path,
            "options": get_options_for_json(expected_options),
            "tokenizer": {
                "class": model.tokenizer.class_name,
                "path": model.tokenizer.download_path,
                "options": get_options_for_json(expected_options),
            }
        }

        # Execute
        result = model.process(models_path='path/to/model',
                               only_configuration=True)

        # Assert
        mock_process_access_token.assert_called_once()
        mock_build_paths.assert_called_once()
        mock_set_class_names.assert_called_once()
        mock_belongs_to_module.assert_called_once()
        mock_download.assert_not_called()
        self.assertEqual(expected_result, json.loads(result))

    @patch('downloader.process_options')
    @patch('downloader.process_access_token', return_value="")
    @patch('downloader.set_class_names', return_value=None)
    @patch('downloader.Model.download', return_value=None)
    @patch('downloader.Model.build_paths', return_value=None)
    @patch('downloader.Model.belongs_to_module', return_value=True)
    def test_process_with_download(
            self, mock_belongs_to_module, mock_build_paths,
            mock_download, mock_set_class_names, mock_process_access_token,
            mock_process_options
    ):
        # Options
        input_options = ["key='test'"]
        expected_options = {"key": "\"test\""}

        # init
        model = Model(name="TestModel", module=TRANSFORMERS,
                      class_name="TestClass", options=input_options)
        model.tokenizer = Tokenizer(class_name="TokenizerClass",
                                    options=input_options)

        # Options
        mock_process_options.return_value = expected_options

        # Prepare
        expected_result = {
            "module": model.module,
            "class": model.class_name,
            "path": model.download_path,
            "options": get_options_for_json(expected_options),
            "tokenizer": {
                "class": model.tokenizer.class_name,
                "path": model.tokenizer.download_path,
                "options": get_options_for_json(expected_options),
            }
        }

        # Execute
        result = model.process(models_path='path/to/model')

        # Assert
        mock_process_access_token.assert_called_once()
        mock_process_options.assert_called()
        mock_build_paths.assert_called_once()
        mock_set_class_names.assert_called_once()
        mock_belongs_to_module.assert_called_once()
        mock_download.assert_called_once()
        self.assertEqual(expected_result, json.loads(result))

    @patch('downloader.download_model')
    def test_download_model_skip_model_bad_module(self, mock_download_model):
        # Init
        model = Model(name="TestModel", module="other")
        model.download_path = "path/to/model"
        model.class_name = "class_name"
        model.options = ["test='test'"]
        model.validate = MagicMock()
        model.build_paths = MagicMock()

        # Execute
        model.download(overwrite=True, skip=DOWNLOAD_MODEL,
                       options={"test": "\"test\""}, options_tokenizer={},
                       access_token=None)

        # Assert
        mock_download_model.assert_called_once()

    @patch('downloader.download_model')
    def test_download_model_skip_model_correct_module(self,
                                                      mock_download_model):
        # Init
        model = Model(name="TestModel", module=TRANSFORMERS)
        model.download_path = "path/to/model"
        model.class_name = "class_name"
        model.options = ["test='test'"]
        model.validate = MagicMock()
        model.build_paths = MagicMock()

        # Execute
        model.download(overwrite=True, skip=DOWNLOAD_MODEL,
                       options={"test": "\"test\""}, options_tokenizer={},
                       access_token=None)

        # Assert
        mock_download_model.assert_called_once()

    @patch('downloader.download_transformers_tokenizer')
    def test_download_model_skip_model_no_module(self,
                                                 mock_download_tokenizer):
        # Init
        model = Model(name="TestModel", module='')
        model.download_path = "path/to/model"
        model.options = ["test='test'"]
        model.validate = MagicMock()
        model.build_paths = MagicMock()

        # Execute
        model.download(overwrite=True, skip=DOWNLOAD_MODEL,
                       options={"test": "\"test\""}, options_tokenizer={},
                       access_token=None)

        # Assert
        mock_download_tokenizer.assert_called_once()

    @patch('downloader.download_transformers_tokenizer')
    @patch('downloader.process_options')
    @patch('downloader.process_access_token', return_value="")
    @patch('downloader.set_class_names', return_value=None)
    @patch('downloader.Model.download', return_value=None)
    @patch('downloader.Model.build_paths', return_value=None)
    def test_process_skip_model_no_module(
            self, mock_build_paths,
            mock_download, mock_set_class_names, mock_process_access_token,
            mock_process_options,
            mock_tokenizer_downloader
    ):
        # Options
        input_options = ["key='test'"]
        expected_options = {"key": "\"test\""}

        # init
        model = Model(name="TestModel", module='',
                      class_name="TestClass", options=input_options)
        model.tokenizer = Tokenizer(class_name="TokenizerClass",
                                    options=input_options)

        # Options
        mock_process_options.return_value = expected_options
        # Execute
        model.process(models_path='path/to/model', skip=DOWNLOAD_MODEL)

        # Assert
        mock_process_access_token.assert_called_once()
        mock_process_options.assert_called()
        mock_build_paths.assert_called_once()
        mock_set_class_names.assert_called_once()
        mock_download.assert_called_once()
        mock_tokenizer_downloader.assert_called_once()

    @patch('downloader.download_transformers_tokenizer')
    @patch('downloader.process_options')
    @patch('downloader.process_access_token', return_value="")
    @patch('downloader.set_class_names', return_value=None)
    @patch('downloader.Model.download', return_value=None)
    @patch('downloader.Model.build_paths', return_value=None)
    def test_process_skip_model_bad_module(
            self, mock_build_paths,
            mock_download, mock_set_class_names, mock_process_access_token,
            mock_process_options,
            mock_tokenizer_downloader
    ):
        # Options
        input_options = ["key='test'"]
        expected_options = {"key": "\"test\""}

        # init
        model = Model(name="TestModel", module='other',
                      class_name="TestClass", options=input_options)
        model.tokenizer = Tokenizer(class_name="TokenizerClass",
                                    options=input_options)

        # Options
        mock_process_options.return_value = expected_options
        # Execute
        model.process(models_path='path/to/model', skip=DOWNLOAD_MODEL)

        # Assert
        mock_process_access_token.assert_called_once()
        mock_process_options.assert_called()
        mock_build_paths.assert_called_once()
        mock_set_class_names.assert_called_once()
        mock_download.assert_called_once()
        mock_tokenizer_downloader.assert_called_once()

    @patch('downloader.download_transformers_tokenizer')
    @patch('downloader.process_options')
    @patch('downloader.process_access_token', return_value="")
    @patch('downloader.set_class_names', return_value=None)
    @patch('downloader.Model.download', return_value=None)
    @patch('downloader.Model.build_paths', return_value=None)
    def test_process_skip_model_correct_module(
            self, mock_build_paths,
            mock_download, mock_set_class_names, mock_process_access_token,
            mock_process_options,
            mock_tokenizer_downloader
    ):
        # Options
        input_options = ["key='test'"]
        expected_options = {"key": "\"test\""}

        # init
        model = Model(name="TestModel", module=TRANSFORMERS,
                      class_name="TestClass", options=input_options)
        model.tokenizer = Tokenizer(class_name="TokenizerClass",
                                    options=input_options)

        # Options
        mock_process_options.return_value = expected_options
        # Execute
        model.process(models_path='path/to/model', skip=DOWNLOAD_MODEL)

        # Assert
        mock_process_access_token.assert_called_once()
        mock_process_options.assert_called()
        mock_build_paths.assert_called_once()
        mock_set_class_names.assert_called_once()
        mock_download.assert_called_once()
        mock_tokenizer_downloader.assert_called_once()


if __name__ == '__main__':
    unittest.main()  # pragma: no cover
