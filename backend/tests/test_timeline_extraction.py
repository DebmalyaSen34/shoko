import os
import shutil
import tempfile
import unittest
from unittest import mock
from src.workflows.timeline_extraction import extract_timeline_from_project

class TestTimelineExtractionWorkflow(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for output files
        self.test_dir = tempfile.mkdtemp()
        self.output_base_dir = os.path.join(self.test_dir, "data")

    def tearDown(self):
        # Clean up temporary test files
        shutil.rmtree(self.test_dir)

    @mock.patch("src.workflows.timeline_extraction.parse_prproj_to_json")
    def test_successful_timeline_extraction(self, mock_parse):
        # Arrange
        # Create a dummy prproj file so existence check passes
        dummy_prproj = os.path.join(self.test_dir, "test_project.prproj")
        with open(dummy_prproj, "w") as f:
            f.write("dummy data")

        project_name = "test_project_1"

        # Side effect to simulate output creation by the parser
        def side_effect(prproj_path, output_json_path):
            with open(output_json_path, "w") as f:
                f.write('{"timeline": "mocked"}')

        mock_parse.side_effect = side_effect

        # Act
        timeline_path = extract_timeline_from_project(
            prproj_path=dummy_prproj,
            project_name=project_name,
            output_base_dir=self.output_base_dir
        )

        # Assert
        expected_path = os.path.abspath(
            os.path.join(self.output_base_dir, project_name, "timeline.json")
        )
        self.assertEqual(timeline_path, expected_path)
        self.assertTrue(os.path.exists(timeline_path))
        
        # Verify mocked parser was called with correct arguments
        mock_parse.assert_called_once_with(
            os.path.abspath(dummy_prproj),
            expected_path
        )

        # Verify content
        with open(timeline_path, "r") as f:
            content = f.read()
        self.assertEqual(content, '{"timeline": "mocked"}')

    def test_missing_prproj_file(self):
        # Arrange
        non_existent_prproj = os.path.join(self.test_dir, "missing.prproj")

        # Act & Assert
        with self.assertRaises(FileNotFoundError):
            extract_timeline_from_project(
                prproj_path=non_existent_prproj,
                project_name="my_project",
                output_base_dir=self.output_base_dir
            )

if __name__ == "__main__":
    unittest.main()
