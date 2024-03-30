import unittest
import torch
from unittest.mock import MagicMock, patch
from sdk import ModelTransformers, Devices


class TestModelTransformers(unittest.TestCase):
    def setUp(self):
        # Set up mock objects and parameters
        self.model_name = "test_model"
        self.model_path = "/path/to/model"
        self.tokenizer_path = "/path/to/tokenizer"
        self.task = "text-generation"
        self.model_class_mock = MagicMock()
        self.tokenizer_class_mock = MagicMock()
        self.device = "cpu"
        self.kwargs = {"param1": "value1", "param2": "value2"}
        self.transformers_pipeline = MagicMock()

    @patch("sdk.ModelTransformers.create_pipeline")
    def test_init(self, create_pipeline_mock):
        # Arrange
        instance = ModelTransformers(
            self.model_name, self.model_path, self.tokenizer_path,
            self.task, self.model_class_mock,
            self.tokenizer_class_mock, self.device
        )

        # Assert
        self.assertEqual(instance.model_name, self.model_name)
        self.assertEqual(instance.model_path, self.model_path)
        self.assertEqual(instance.tokenizer_path, self.tokenizer_path)
        self.assertEqual(instance.task, self.task)
        self.assertEqual(instance.model_class, self.model_class_mock)
        self.assertEqual(instance.tokenizer_class,
                         self.tokenizer_class_mock)
        self.assertEqual(instance.device, self.device)
        self.assertFalse(instance.loaded)
        create_pipeline_mock.assert_called_once()

    @patch("sdk.ModelTransformers.create_pipeline")
    def test_load_model_when_loaded(self, create_pipeline_mock):
        # Arrange
        instance = ModelTransformers(
            self.model_name, self.model_path, self.tokenizer_path,
            self.task, self.model_class_mock,
            self.tokenizer_class_mock, self.device
        )
        instance.loaded = True

        # Act
        result = instance.load_model()

        # Assert
        self.assertTrue(result)
        self.assertTrue(instance.loaded)

    @patch("sdk.ModelTransformers.create_pipeline")
    def test_load_model_when_device_meta(self, create_pipeline_mock):
        # Arrange
        instance = ModelTransformers(
            self.model_name, self.model_path, self.tokenizer_path,
            self.task, self.model_class_mock,
            self.tokenizer_class_mock, self.device
        )
        instance.loaded = False
        instance.transformers_pipeline = MagicMock()
        instance.transformers_pipeline.device = torch.device(Devices.RESET.value)

        # Act
        result = instance.load_model()

        # Assert
        self.assertTrue(result)
        self.assertTrue(instance.loaded)

    @patch("sdk.ModelTransformers.create_pipeline")
    def test_load_model_on_reset_device(self, create_pipeline_mock):
        # Arrange
        instance = ModelTransformers(
            self.model_name, self.model_path, self.tokenizer_path,
            self.task, self.model_class_mock,
            self.tokenizer_class_mock, Devices.RESET.value
        )

        # Act
        result = instance.load_model()

        # Assert
        self.assertFalse(result)
        self.assertFalse(instance.loaded)

    @patch("sdk.ModelTransformers.create_pipeline")
    def test_load_model_successfully(self, create_pipeline_mock):
        # Arrange
        pipeline_mock = MagicMock()
        create_pipeline_mock.return_value = None
        instance = ModelTransformers(
            self.model_name, self.model_path, self.tokenizer_path,
            self.task, self.model_class_mock,
            self.tokenizer_class_mock, self.device
        )
        instance.transformers_pipeline = pipeline_mock

        # Act
        result = instance.load_model()

        # Assert
        self.assertTrue(result)
        self.assertTrue(instance.loaded)
        pipeline_mock.model.to.assert_called_once_with(device='cpu')

    @patch("torch.cuda.empty_cache")
    @patch("torch.cuda.ipc_collect")
    @patch("sdk.ModelTransformers.create_pipeline")
    def test_unload_model_when_loaded(self, create_pipeline_mock,
                                      mock_ipc_collect, mock_empty_cache):
        # Arrange
        pipeline_mock = MagicMock()
        instance = ModelTransformers(
            self.model_name, self.model_path, self.tokenizer_path,
            self.task, self.model_class_mock,
            self.tokenizer_class_mock, self.device
        )
        instance.loaded = True
        instance.transformers_pipeline = MagicMock(
            return_value=pipeline_mock
        )

        # Act
        result = instance.unload_model()

        # Assert
        self.assertTrue(result)
        self.assertFalse(instance.loaded)
        mock_ipc_collect.assert_called_once()
        mock_empty_cache.assert_called_once()

    @patch("sdk.ModelTransformers.create_pipeline")
    def test_unload_model_when_not_loaded(self, create_pipeline_mock):
        # Arrange
        instance = ModelTransformers(
            self.model_name, self.model_path, self.tokenizer_path,
            self.task, self.model_class_mock,
            self.tokenizer_class_mock, self.device
        )
        instance.loaded = False

        # Act
        result = instance.unload_model()

        # Assert
        self.assertFalse(result)
        self.assertFalse(instance.loaded)

    @patch("sdk.ModelTransformers.create_pipeline")
    def test_generate_prompt(self, create_pipeline_mock):
        # Arrange
        instance = ModelTransformers(
            self.model_name, self.model_path, self.tokenizer_path,
            self.task, self.model_class_mock,
            self.tokenizer_class_mock, self.device
        )
        instance.transformers_pipeline = MagicMock()

        # Act
        result = instance.generate_prompt("test_prompt")

        # Assert
        self.assertEqual(result, instance.transformers_pipeline.return_value)

    # def test_create_pipeline_loaded(self):
    #     # Arrange
    #     instance = ModelTransformers(
    #         self.model_name, self.model_path, self.tokenizer_path,
    #         self.task, pipeline, self.tokenizer_class_mock, self.device
    #     )
    #     # Assert
    #     self.assertIsNotNone(instance.transformers_pipeline)
    # def test_create_pipeline(self):
    #     # Arrange
    #     instance = ModelTransformers(
    #         self.model_name, self.model_path, self.tokenizer_path,
    #         self.task, pipeline, self.tokenizer_class_mock, self.device
    #     )
    #
    #     # Assert
    #     self.assertIsNotNone(instance.transformers_pipeline)

    # @patch("transformers.pipeline")
    # def test_create_pipeline_successfully(self, mock_pipeline):
    #
    #     # Act
    #     instance = ModelTransformers(
    #         self.model_name, self.model_path, self.tokenizer_path,
    #         self.task, pipeline_mock, self.tokenizer_class_mock, self.device,
    #     )
    #     # Assert
    #     self.assertIsNotNone(instance.tokenizer_pipeline)

    @patch("sdk.ModelTransformers.create_pipeline")
    def test_set_model_pipeline_args(self, mock):
        instance = ModelTransformers(
            self.model_name, self.model_path, self.tokenizer_path,
            self.task, self.model_class_mock,
            self.tokenizer_class_mock, self.device
        )
        # Arrange
        kwargs = {"arg1": "value1", "arg2": "value2"}

        # Act
        instance.set_model_pipeline_args(**kwargs)

        # Assert
        self.assertEqual(instance.model_pipeline_args, kwargs)

    @patch("sdk.ModelTransformers.create_pipeline")
    def test_set_model_pipeline_args_empty(self, mock):
        instance = ModelTransformers(
            self.model_name, self.model_path, self.tokenizer_path,
            self.task, self.model_class_mock,
            self.tokenizer_class_mock, self.device
        )
        # Arrange
        initial_args = {"initial_arg": "initial_value"}
        instance.model_pipeline_args = initial_args.copy()

        # Act
        instance.set_model_pipeline_args()

        # Assert
        self.assertEqual(instance.model_pipeline_args, initial_args)

    @patch("sdk.ModelTransformers.create_pipeline")
    def test_set_tokenizer_pipeline_args(self, mock):
        instance = ModelTransformers(
            self.model_name, self.model_path, self.tokenizer_path,
            self.task, self.model_class_mock,
            self.tokenizer_class_mock, self.device
        )
        # Arrange
        kwargs = {"arg1": "value1", "arg2": "value2"}

        # Act
        instance.set_tokenizer_pipeline_args(**kwargs)

        # Assert
        self.assertEqual(instance.tokenizer_pipeline_args, kwargs)

    @patch("sdk.ModelTransformers.create_pipeline")
    def test_set_tokenizer_pipeline_args_empty(self, mock):
        instance = ModelTransformers(
            self.model_name, self.model_path, self.tokenizer_path,
            self.task, self.model_class_mock,
            self.tokenizer_class_mock, self.device
        )
        # Arrange
        initial_args = {"initial_arg": "initial_value"}
        instance.tokenizer_pipeline_args = initial_args.copy()

        # Act
        instance.set_tokenizer_pipeline_args()

        # Assert
        self.assertEqual(instance.tokenizer_pipeline_args, initial_args)


if __name__ == "__main__":
    unittest.main()
