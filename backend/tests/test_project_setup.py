import os
import shutil
import tempfile
import unittest
import zipfile
from src.workflows.project_setup import setup_project_workspace

class TestProjectSetupWorkflow(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for our test files
        self.test_dir = tempfile.mkdtemp()
        self.zip_path = os.path.join(self.test_dir, "project.zip")
        self.assets_dir = os.path.join(self.test_dir, "assets")

    def tearDown(self):
        # Clean up temporary test files
        shutil.rmtree(self.test_dir)

    def _create_mock_zip(self, file_contents: dict[str, str]):
        """Helper to create a zip file with specified filenames and dummy contents."""
        with zipfile.ZipFile(self.zip_path, 'w') as zf:
            for filepath, content in file_contents.items():
                zf.writestr(filepath, content)

    def test_successful_setup(self):
        # Arrange
        mock_files = {
            "my_project/my_timeline.prproj": "dummy prproj data",
            "my_project/videos/clip1.mp4": "video content 1",
            "my_project/videos/clip2.mov": "video content 2",
            "my_project/audio/song.mp3": "audio content 1",
            "my_project/audio/effect.wav": "audio content 2",
            "my_project/notes.txt": "some text notes that should be ignored",
        }
        self._create_mock_zip(mock_files)
        project_name = "test_project"

        # Act
        project_dir, prproj_path = setup_project_workspace(
            zip_path=self.zip_path,
            project_name=project_name,
            assets_dir=self.assets_dir
        )

        # Assert
        expected_project_dir = os.path.join(self.assets_dir, project_name)
        self.assertEqual(project_dir, os.path.abspath(expected_project_dir))

        expected_prproj_path = os.path.join(expected_project_dir, "my_timeline.prproj")
        self.assertEqual(prproj_path, os.path.abspath(expected_prproj_path))

        # Check directories exist
        subdirs = [
            "00_style", "01_characters", "02_props", "03_locations",
            "04_audio", "05_references", "06_clips/_final", "06_clips/_raw"
        ]
        for subdir in subdirs:
            self.assertTrue(os.path.isdir(os.path.join(expected_project_dir, subdir)))

        # Verify moved files
        self.assertTrue(os.path.exists(prproj_path))
        self.assertTrue(os.path.exists(os.path.join(expected_project_dir, "06_clips/_raw/clip1.mp4")))
        self.assertTrue(os.path.exists(os.path.join(expected_project_dir, "06_clips/_raw/clip2.mov")))
        self.assertTrue(os.path.exists(os.path.join(expected_project_dir, "04_audio/song.mp3")))
        self.assertTrue(os.path.exists(os.path.join(expected_project_dir, "04_audio/effect.wav")))

        # Verify ignored/unmoved file
        self.assertFalse(os.path.exists(os.path.join(expected_project_dir, "notes.txt")))
        self.assertFalse(os.path.exists(os.path.join(expected_project_dir, "my_project/notes.txt")))

    def test_missing_zip_file(self):
        # Arrange
        non_existent_zip = os.path.join(self.test_dir, "does_not_exist.zip")

        # Act & Assert
        with self.assertRaises(FileNotFoundError):
            setup_project_workspace(
                zip_path=non_existent_zip,
                project_name="my_project",
                assets_dir=self.assets_dir
            )

    def test_missing_prproj_file(self):
        # Arrange
        mock_files = {
            "clip1.mp4": "video content 1",
            "song.mp3": "audio content 1",
        }
        self._create_mock_zip(mock_files)

        # Act & Assert
        with self.assertRaisesRegex(ValueError, "No .prproj file found in the zip archive"):
            setup_project_workspace(
                zip_path=self.zip_path,
                project_name="my_project",
                assets_dir=self.assets_dir
            )

    def test_case_insensitive_extensions(self):
        # Arrange
        mock_files = {
            "PROJECT.PRPROJ": "dummy",
            "CLIP.MP4": "video",
            "SONG.WAV": "audio",
        }
        self._create_mock_zip(mock_files)
        project_name = "test_project_case"

        # Act
        project_dir, prproj_path = setup_project_workspace(
            zip_path=self.zip_path,
            project_name=project_name,
            assets_dir=self.assets_dir
        )

        # Assert
        self.assertTrue(os.path.exists(prproj_path))
        self.assertTrue(os.path.exists(os.path.join(project_dir, "06_clips/_raw/CLIP.MP4")))
        self.assertTrue(os.path.exists(os.path.join(project_dir, "04_audio/SONG.WAV")))

if __name__ == "__main__":
    unittest.main()
