import unittest
import os
import sys
import argparse
import json
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..")))
from downloader import (Model, Tokenizer, download_model,  # noqa: E402
                        download_transformers_tokenizer,
                        is_path_valid_for_download, process_options,
                        map_args_to_model, main, exit_error)


class TestDownloader(unittest.TestCase):

    def test_model_validation_valid(self):
        valid_model = Model(name="valid_model", module="some_module")
        valid_model.validate()

    def test_model_validation_invalid(self):
        invalid_model = Model(name="", module="some_module")
        with self.assertRaises(SystemExit) as context:
            invalid_model.validate()
        self.assertEqual(context.exception.code, 1)

    def test_is_transformers_true(self):
        transformers_model = Model(name="transformer_model",
                                   module="transformers")
        self.assertTrue(transformers_model.is_transformers())

    def test_is_transformers_false(self):
        non_transformers_model = Model(name="non_transformer_model",
                                       module="diffusers")
        self.assertFalse(non_transformers_model.is_transformers())

    def test_build_paths(self):
        model = Model(name="test_model", module="some_module")
        models_path = "/models"
        model.build_paths(models_path)
        self.assertEqual(model.base_path, os.path.join(models_path,
                                                       "test_model"))
        self.assertEqual(model.download_path, os.path.join(models_path,
                                                           "test_model"))

    def test_build_paths_transformers(self):
        model = Model(name="TestModel", module="transformers")
        model.build_paths("/models")
        expected_download_path = "/models\\TestModel\\model"
        self.assertEqual(model.download_path, expected_download_path)

    def test_build_paths_non_transformers(self):
        model = Model(name="TestModel", module="non_transformers")
        model.build_paths("/models")
        expected_download_path = "/models\\TestModel"
        self.assertEqual(model.download_path, expected_download_path)

    @patch('os.path.exists')
    @patch('os.listdir')
    def test_is_path_valid_for_download(self, mock_exists, mock_listdir):
        mock_exists.return_value = True
        mock_listdir.return_value = ["file1", "file2"]
        self.assertTrue(is_path_valid_for_download("/models", False))
        mock_exists.assert_called_once_with("/models")
        mock_listdir.assert_called_once_with("/models")

    def test_process_options(self):
        options_list = ["key1='value1'", "key2='value2'", "key3=3"]
        options_dict = process_options(options_list)
        self.assertEqual(options_dict, {"key1": "value1", "key2": "value2",
                                        "key3": 3})

    def test_process_options_invalid_format(self):
        options_list = ["invalid_option"]
        with self.assertRaises(SystemExit) as context:
            process_options(options_list)
        self.assertEqual(context.exception.code, 1)

    @patch('builtins.eval', MagicMock(return_value="evaluated_value"))
    def test_process_options_evaluated_value(self):
        options_list = ["key=expression"]
        options_dict = process_options(options_list)
        self.assertEqual(options_dict, {"key": "evaluated_value"})

    @patch('builtins.eval', side_effect=Exception("Evaluation failed"))
    def test_process_options_evaluated_value_error(
            self, mock_eval):  # noqa: F841
        options_list = ["key=expression"]
        with self.assertRaises(SystemExit) as context:
            process_options(options_list)
        self.assertEqual(context.exception.code, 1)

    @patch('importlib.import_module', MagicMock(return_value=MagicMock()))
    def test_process_options_import_module(self):
        options_dict = process_options(["key=module.attribute"])
        self.assertIn("key", options_dict)
        self.assertIsInstance(options_dict["key"], MagicMock)

    @patch('importlib.import_module', side_effect=ImportError("Import failed"))
    def test_process_options_import_module_error(
            self, mock_import_module):  # noqa: F841
        options_list = ["key=module.attribute"]
        with self.assertRaises(SystemExit) as context:
            process_options(options_list)
        self.assertEqual(context.exception.code, 1)

    @patch('builtins.print')
    @patch('sys.exit')
    def test_exit_error(self, mock_exit, mock_print):
        exit_error("Test message", 42)
        mock_print.assert_called_once_with("Test message", file=sys.stderr)
        mock_exit.assert_called_once_with(42)

    @patch('builtins.print')
    @patch('sys.exit')
    def test_exit_error_default_code(self, mock_exit, mock_print):
        exit_error("Test message")
        mock_print.assert_called_once_with("Test message", file=sys.stderr)
        mock_exit.assert_called_once_with(1)

    @patch('builtins.print')
    @patch('sys.exit')
    def test_exit_error_default_code_empty(self, mock_exit, mock_print):
        exit_error("")
        mock_print.assert_called_once_with("", file=sys.stderr)
        mock_exit.assert_called_once_with(1)

    def test_map_args_to_model(self):
        args = argparse.Namespace(
            models_path="/models",
            model_name="test_model",
            model_module="some_module",
            model_class="TestClass",
            model_options=["key1=value1"],
            tokenizer_class="TestTokenizer",
            tokenizer_options=["key2=value2"],
            overwrite=False,
            skip="tokenizer",
            emf_client=False
        )
        model = map_args_to_model(args)
        self.assertEqual(model.name, "test_model")
        self.assertEqual(model.module, "some_module")
        self.assertEqual(model.class_name, "TestClass")
        self.assertEqual(model.options, ["key1=value1"])
        self.assertIsInstance(model.tokenizer, Tokenizer)
        self.assertEqual(model.tokenizer.class_name, "TestTokenizer")
        self.assertEqual(model.tokenizer.options, ["key2=value2"])

    @patch('downloader.Model.download')
    @patch('builtins.print')
    def test_main(self, mock_print, mock_model_download):
        args = argparse.Namespace(
            models_path="/models",
            model_name="test_model",
            model_module="some_module",
            model_class="TestClass",
            model_options=["key1=value1"],
            tokenizer_class="TestTokenizer",
            tokenizer_options=["key2=value2"],
            overwrite=False,
            skip="tokenizer",
            emf_client=False
        )
        with patch('sys.argv', ['script_name']):
            with patch('argparse.ArgumentParser.parse_args',
                       return_value=args):
                main()
                mock_model_download.assert_called_once_with("/models",
                                                            "tokenizer", False)
                mock_print.assert_not_called()

    @patch('downloader.Model.download')
    @patch('builtins.print')
    def test_main_emf_client(self, mock_print, mock_model_download):
        args = argparse.Namespace(
            models_path="/models",
            model_name="test_model",
            model_module="some_module",
            model_class="TestClass",
            model_options=["key1=value1"],
            tokenizer_class="TestTokenizer",
            tokenizer_options=["key2=value2"],
            overwrite=False,
            skip="tokenizer",
            emf_client=True
        )
        with patch('sys.argv', ['script_name']):
            with patch('argparse.ArgumentParser.parse_args',
                       return_value=args):
                main()
                mock_model_download.assert_called_once_with("/models",
                                                            "tokenizer", False)
                mock_print.assert_called_once()

    def test_download_with_skip_model(self):
        model = Model(name="TestModel", module="transformers",
                      class_name="AutoModel", options=["key1=value1"])
        model.validate = MagicMock()
        model.build_paths = MagicMock()
        model.is_transformers = MagicMock(return_value=False)

        with patch('downloader.is_path_valid_for_download',
                   return_value=False):
            with patch('downloader.exit_error'):
                with patch('downloader.process_options', return_value={}):
                    download_model(model, overwrite=False)

        with patch('json.dumps', return_value='{}'):
            result = model.download("/models", skip="model",
                                    overwrite=False)

        model.validate.assert_called_once()
        model.build_paths.assert_called_once()
        self.assertEqual(result, '{}')

    @patch('os.path.exists', return_value=True)
    @patch('os.listdir', return_value=['file1', 'file2'])
    @patch('sys.exit', side_effect=SystemExit)
    def test_download_model_path(
            self, mock_exit, mock_listdir, mock_exists):  # noqa: F841
        model = Model(name='TestModel', module='transformers',
                      class_name='CustomClass')
        overwrite = False

        with self.assertRaises(SystemExit) as context:
            download_model(model, overwrite)
        self.assertEqual(context.exception.code, None)

    def test_download_with_skip_model_two(self):
        model = Model(name="TestModel", module="transformers",
                      class_name=None, options=["key1=value1"])
        model.validate = MagicMock()
        model.path = "/model"
        model.build_paths = MagicMock()
        model.is_transformers = MagicMock(return_value=False)

        with patch('downloader.is_path_valid_for_download',
                   return_value=False):
            with patch('downloader.exit_error'):
                with patch('downloader.process_options', return_value={}):
                    download_model(model, overwrite=False)

    def test_download_transformers_tokenizer_exists(self):
        model = Model(name="TestModel", module="transformers",
                      tokenizer=Tokenizer(class_name="AutoTokenizer"))
        model.tokenizer.download_path = "/AutoTokenizer"

        with self.assertRaises(SystemExit) as context:
            download_transformers_tokenizer(model, overwrite=False)

        self.assertEqual(context.exception.code, 3)

    @patch('downloader.download_model')
    def test_download(self, mock_download_model):
        self.model_object = Model(name="example_model", module="diffusers")
        result = self.model_object.download(  # noqa: F841
            models_path="/models", overwrite=True)
        mock_download_model.assert_called_once_with(self.model_object, True)
        result = self.model_object.download(models_path="/models",
                                            overwrite=True, skip="")

        expected_result = {
            "path": self.model_object.download_path,
            "module": self.model_object.module,
            "class": self.model_object.class_name
        }
        self.assertEqual(json.loads(result), expected_result)

    @patch('os.path.exists')
    @patch('os.listdir')
    def test_download_transformers_tokenizer_exception(self, mock_listdir,
                                                       mock_exists):
        mock_exists.return_value = False
        mock_listdir.return_value = []
        model = Model(name="TestModel", module="transformers",
                      tokenizer=Tokenizer(
                          class_name="PreTrainedTokenizerFast"))
        model.tokenizer.download_path = "/tokenizer_path"
        model.base_path = "/model_path"

        with self.assertRaises(SystemExit) as context:
            download_transformers_tokenizer(model, overwrite=False)

        self.assertEqual(context.exception.code, 3)


if __name__ == '__main__':
    unittest.main()  # pragma: no cover
