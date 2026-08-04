import os
import json
import shutil
import tempfile
import unittest
from unittest import mock

from src.workflows.video_generation import run_video_generation_workflow

class TestVideoGenerationWorkflow(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.prompts_path = os.path.join(self.test_dir, "video_prompts.json")
        self.output_dir = os.path.join(self.test_dir, "seedance_videos")

        # Mock video_prompts.json content
        self.mock_prompts = [
            {
                "clip_used": "clip1.mp4",
                "generation_type": "none",
                "video_model_prompt": "",
                "status": "none"
            },
            {
                "clip_used": "clip2.mp4",
                "category": "video",
                "generation_type": "complex",
                "video_model_prompt": "Warm sunlit hall... Vir enters playfully... mouth moving in sync.",
                "selected_assets": ["/dummy/vir-sheet.png"],
                "clip_frame_paths": ["/dummy/frame1.jpg"],
                "audio_used": "audio1.mp3",
                "audio_path": "/dummy/audio1.mp3",
                "audio_url": "data:audio/mp3;base64,dummy_audio",
                "is_dialogue_active": True,
                "generate_audio": True,
                "ratio": "9:16",
                "duration": 5,
                "status": "success",
                "first_frame_url": "data:image/png;base64,dummy_first_frame",
                "initial_frame_image_path": "/dummy/last_frame.jpg"
            }
        ]

        with open(self.prompts_path, "w", encoding="utf-8") as f:
            json.dump(self.mock_prompts, f)

        # Mock workspace directory structure
        self.workspace_dir = os.path.join(self.test_dir, "workspace")
        os.makedirs(os.path.join(self.workspace_dir, "assets", "test_video_project", "06_clips", "_final"), exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @mock.patch("src.workflows.video_generation.create_asset_url_cache")
    @mock.patch("src.workflows.video_generation.build_segmind_payload")
    @mock.patch("src.workflows.video_generation.create_seedance_task")
    @mock.patch("src.workflows.video_generation.save_video_bytes")
    @mock.patch("os.rename")
    @mock.patch("os.makedirs")
    def test_run_video_generation_workflow_successful(
        self, mock_makedirs, mock_rename, mock_save, mock_create, mock_build_payload, mock_cache_factory
    ):
        # Arrange
        mock_cache = mock.Mock()
        mock_cache_factory.return_value = mock_cache
        mock_build_payload.return_value = {
            "prompt": "mock prompt",
            "duration": 5,
            "aspect_ratio": "9:16",
            "generate_audio": True,
        }
        mock_create.return_value = {
            "id": "segmind_999",
            "content": {"bytes": b"video-bytes"},
        }

        # Act
        previous_cwd = os.getcwd()
        os.chdir(self.workspace_dir)
        try:
            generated_clips = run_video_generation_workflow(
                prompts_json_path=self.prompts_path,
                project_name="test_video_project",
                segmind_api_key="mock_api_key",
                output_dir=self.output_dir,
                poll_interval_seconds=1
            )
        finally:
            os.chdir(previous_cwd)

        # Assert
        # Verify calls
        mock_build_payload.assert_called_once()
        self.assertEqual(mock_build_payload.call_args.kwargs["api_key"], "mock_api_key")
        self.assertEqual(mock_build_payload.call_args.kwargs["cache"], mock_cache)
        mock_create.assert_called_once_with(
            api_key="mock_api_key",
            payload={
                "prompt": "mock prompt",
                "duration": 5,
                "aspect_ratio": "9:16",
                "generate_audio": True,
            },
        )
        mock_save.assert_called_once_with(
            b"video-bytes",
            os.path.join(self.output_dir, "segmind_999.mp4")
        )

        # Verify output list
        self.assertEqual(len(generated_clips), 1)
        expected_final_path = os.path.abspath(
            os.path.join(self.workspace_dir, "assets", "test_video_project", "06_clips", "_final", "clip2.mp4")
        )
        self.assertEqual(os.path.realpath(generated_clips[0]), os.path.realpath(expected_final_path))

    def test_missing_api_key(self):
        # Act & Assert
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                run_video_generation_workflow(
                    prompts_json_path=self.prompts_path,
                    project_name="test_video_project",
                    segmind_api_key=None,
                    ark_api_key=None
                )

    def test_missing_prompts_file(self):
        # Act & Assert
        with self.assertRaises(FileNotFoundError):
            run_video_generation_workflow(
                prompts_json_path=os.path.join(self.test_dir, "missing.json"),
                project_name="test_video_project",
                segmind_api_key="mock_key"
            )

if __name__ == "__main__":
    unittest.main()
