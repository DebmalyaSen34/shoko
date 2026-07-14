import base64
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.copy_image import (
    _file_data_url,
    _response_value,
    download_image,
    save_b64_image,
    main,
)


class CopyImageScriptTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.input_path = Path(self.temp_dir.name) / "input.png"
        self.output_path = Path(self.temp_dir.name) / "output.png"

        # Create dummy input image file
        with open(self.input_path, "wb") as f:
            f.write(b"png-dummy-bytes")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_file_data_url_encodes_png_correctly(self):
        url = _file_data_url(self.input_path)
        expected_prefix = "data:image/png;base64,"
        self.assertTrue(url.startswith(expected_prefix))
        encoded_data = url[len(expected_prefix):]
        self.assertEqual(b"png-dummy-bytes", base64.b64decode(encoded_data))

    def test_response_value_extracts_from_dict_and_object(self):
        # Dictionary
        self.assertEqual("val1", _response_value({"key1": "val1"}, "key1"))
        self.assertEqual("default_val", _response_value({"key1": "val1"}, "key2", "default_val"))

        # Object
        class TestObj:
            def __init__(self):
                self.key1 = "val1"

        obj = TestObj()
        self.assertEqual("val1", _response_value(obj, "key1"))
        self.assertEqual("default_val", _response_value(obj, "key2", "default_val"))

    @mock.patch("urllib.request.urlopen")
    def test_download_image_writes_file(self, mock_urlopen):
        mock_response = mock.MagicMock()
        mock_response.read.return_value = b"downloaded-image-bytes"
        mock_urlopen.return_value.__enter__.return_value = mock_response

        download_image("http://example.com/image.jpg", self.output_path)

        self.assertTrue(self.output_path.exists())
        with open(self.output_path, "rb") as f:
            self.assertEqual(b"downloaded-image-bytes", f.read())

    def test_get_size_for_model(self):
        from scripts.copy_image import get_size_for_model
        # Seedream models -> 9:16 2K
        self.assertEqual("1600x2848", get_size_for_model("seedream-5-0-lite"))
        self.assertEqual("1600x2848", get_size_for_model("dreamina-seedream-4-5-251128"))
        # Seededit i2i -> adaptive
        self.assertEqual("adaptive", get_size_for_model("seededit-3-0-i2i"))
        self.assertEqual("adaptive", get_size_for_model("dreamina-seededit-3-0-i2i-241028"))
        # Seededit t2i -> 720x1280
        self.assertEqual("720x1280", get_size_for_model("seededit-3-0-t2i"))

    def test_save_b64_image_decodes_and_writes_file(self):
        b64_str = "data:image/png;base64," + base64.b64encode(b"decoded-image-bytes").decode("ascii")
        save_b64_image(b64_str, self.output_path)

        self.assertTrue(self.output_path.exists())
        with open(self.output_path, "rb") as f:
            self.assertEqual(b"decoded-image-bytes", f.read())

    @mock.patch("scripts.copy_image.Ark")
    @mock.patch("scripts.copy_image.download_image")
    def test_main_with_url_response_format_succeeds(self, mock_download, mock_ark):
        # Setup mock API response
        mock_client = mock.MagicMock()
        mock_ark.return_value = mock_client
        mock_image_item = mock.MagicMock()
        mock_image_item.url = "http://example.com/output.jpg"
        mock_response = mock.MagicMock()
        mock_response.data = [mock_image_item]
        mock_client.images.generate.return_value = mock_response

        # Execute main with URL format
        args = [
            str(self.input_path),
            str(self.output_path),
            "--model", "test-model",
            "--api-key", "test-key",
            "--response-format", "url"
        ]

        exit_code = main(args)

        self.assertEqual(0, exit_code)
        mock_client.images.generate.assert_called_once_with(
            model="test-model",
            prompt="recreate this image exactly, one to one copy, preserving all visual elements, style, colors, composition, and subject matter without any changes",
            image=_file_data_url(self.input_path),
            response_format="url",
            watermark=False,
            sequential_image_generation="disabled",
            size="1600x2848"
        )
        mock_download.assert_called_once_with("http://example.com/output.jpg", self.output_path)

    @mock.patch("scripts.copy_image.Ark")
    @mock.patch("scripts.copy_image.save_b64_image")
    def test_main_with_b64_response_format_succeeds(self, mock_save_b64, mock_ark):
        # Setup mock API response
        mock_client = mock.MagicMock()
        mock_ark.return_value = mock_client
        mock_image_item = mock.MagicMock()
        mock_image_item.b64_json = "dummy_base64_data"
        mock_response = mock.MagicMock()
        mock_response.data = [mock_image_item]
        mock_client.images.generate.return_value = mock_response

        # Execute main with b64_json format
        args = [
            str(self.input_path),
            str(self.output_path),
            "--model", "test-model",
            "--api-key", "test-key",
            "--response-format", "b64_json"
        ]

        exit_code = main(args)

        self.assertEqual(0, exit_code)
        mock_client.images.generate.assert_called_once_with(
            model="test-model",
            prompt="recreate this image exactly, one to one copy, preserving all visual elements, style, colors, composition, and subject matter without any changes",
            image=_file_data_url(self.input_path),
            response_format="b64_json",
            watermark=False,
            sequential_image_generation="disabled",
            size="1600x2848"
        )
        mock_save_b64.assert_called_once_with("dummy_base64_data", self.output_path)


if __name__ == "__main__":
    unittest.main()
