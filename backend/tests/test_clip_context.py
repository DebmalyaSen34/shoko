import json
import os
import tempfile
import unittest
from unittest import mock

from src.workflows.clip_context import analyze_clip_context


class TestClipContextWorkflow(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.assets_dir = os.path.join(self.test_dir, "assets")
        self.output_base_dir = os.path.join(self.test_dir, "app-data")
        self.project_name = "project-a"
        self.raw_dir = os.path.join(self.assets_dir, self.project_name, "06_clips", "_raw")
        os.makedirs(self.raw_dir, exist_ok=True)
        self.clip_path = os.path.join(self.raw_dir, "output (23).mp4")
        with open(self.clip_path, "wb") as file:
            file.write(b"video")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.test_dir)

    @mock.patch("src.workflows.clip_context.generate_structured")
    @mock.patch("src.workflows.clip_context.transcribe_audio_segment")
    @mock.patch("src.workflows.clip_context.extract_audio_segment")
    @mock.patch("src.workflows.clip_context.extract_context_frames")
    @mock.patch("src.workflows.clip_context.trim_clip_segment")
    def test_analyzes_timeline_segment_frames_and_saves_under_output_base(
        self, mock_trim, mock_frames, mock_extract_audio, mock_transcribe, mock_generate
    ):
        def trim_side_effect(source_path, output_path, duration_s, source_offset_s=0.0):
            self.assertEqual(self.clip_path, source_path)
            self.assertEqual(4.0, duration_s)
            self.assertEqual(0.0, source_offset_s)
            self.assertIn(os.path.join("app-data", self.project_name, "analysis", "clip_context"), output_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "wb") as file:
                file.write(b"segment")
            return True

        def frames_side_effect(video_path, output_dir, duration_s):
            self.assertTrue(video_path.endswith("segment.mp4"))
            self.assertIn(os.path.join("analysis", "clip_context"), output_dir)
            frame_path = os.path.join(output_dir, "frame_001.jpg")
            os.makedirs(output_dir, exist_ok=True)
            with open(frame_path, "wb") as file:
                file.write(b"frame")
            return [frame_path]

        mock_trim.side_effect = trim_side_effect
        mock_frames.side_effect = frames_side_effect
        audio_path = os.path.join(self.assets_dir, self.project_name, "04_audio", "mix.mp3")
        os.makedirs(os.path.dirname(audio_path), exist_ok=True)
        with open(audio_path, "wb") as file:
            file.write(b"audio")
        mock_extract_audio.side_effect = lambda _input, output, _start, _end: (
            os.makedirs(os.path.dirname(output), exist_ok=True) or open(output, "wb").write(b"audio segment") or True
        )
        mock_transcribe.return_value = "Vir laughs and says hello."
        mock_generate.return_value = {
            "summary": "Mother faces camera without emotion.",
            "visible_characters": ["Mother"],
            "expressions": ["neutral"],
            "gaze": ["looking at camera"],
            "actions": [],
            "blocking": ["Mother centered"],
            "camera_framing": "medium shot",
            "location": "interior",
            "continuity_notes": [],
            "uncertainty_flags": [],
        }
        mock_client = mock.MagicMock()

        result = analyze_clip_context(
            project_name=self.project_name,
            clip_name="output (23).mp4",
            clip_occurrence=7,
            clip_start_s=44.0,
            clip_end_s=48.0,
            clip_duration_s=4.0,
            assets_dir=self.assets_dir,
            output_base_dir=self.output_base_dir,
            client=mock_client,
            provider="openai",
            feedback_items=[{"timestamp": "00:45", "remark": "Show Vir."}],
            audio_name="mix.mp3",
        )

        self.assertEqual("frames_only", result["status"])
        self.assertEqual("transcribed", result["audio_status"])
        self.assertEqual("Vir laughs and says hello.", result["audio_transcript"])
        self.assertTrue(result["audio_segment_path"].endswith("segment_audio.mp3"))
        self.assertTrue(result["clip_context_path"].startswith(os.path.abspath(self.output_base_dir)))
        self.assertNotIn("backend/data", result["clip_context_path"])
        self.assertTrue(os.path.exists(result["clip_context_path"]))
        with open(result["clip_context_path"], "r", encoding="utf-8") as file:
            saved = json.load(file)
        self.assertEqual("Mother faces camera without emotion.", saved["summary"])
        contents = mock_generate.call_args.kwargs["contents"]
        self.assertFalse(any(item.get("type") == "input_file" for item in contents if isinstance(item, dict)))
        self.assertTrue(any(item.get("type") == "input_image" for item in contents if isinstance(item, dict)))
        self.assertIn("AUDIO_TRANSCRIPT: Vir laughs and says hello.", contents[-1])
        mock_client.files.delete.assert_not_called()
        mock_extract_audio.assert_called_once()
        mock_transcribe.assert_called_once()

    @mock.patch("src.workflows.clip_context.generate_structured")
    @mock.patch("src.workflows.clip_context.extract_context_frames")
    @mock.patch("src.workflows.clip_context.trim_clip_segment")
    def test_openai_uses_frames_only_for_clip_analysis(
        self, mock_trim, mock_frames, mock_generate
    ):
        mock_trim.side_effect = lambda _source, output, _duration, source_offset_s=0.0: (
            os.makedirs(os.path.dirname(output), exist_ok=True) or open(output, "wb").write(b"segment") or True
        )
        frame_path = os.path.join(self.test_dir, "frame_001.jpg")
        with open(frame_path, "wb") as file:
            file.write(b"frame")
        mock_frames.return_value = [frame_path]
        mock_generate.return_value = {
            "summary": "Frames show a neutral mother.",
            "visible_characters": ["Mother"],
            "expressions": ["neutral"],
            "gaze": ["toward camera"],
            "actions": [],
            "blocking": [],
            "camera_framing": "",
            "location": None,
            "continuity_notes": [],
            "uncertainty_flags": ["video file was not attached"],
        }

        result = analyze_clip_context(
            project_name=self.project_name,
            clip_name="output (23).mp4",
            clip_occurrence=7,
            clip_start_s=44.0,
            clip_end_s=48.0,
            clip_duration_s=4.0,
            assets_dir=self.assets_dir,
            output_base_dir=self.output_base_dir,
            client=mock.MagicMock(),
            provider="openai",
        )

        self.assertEqual("frames_only", result["status"])
        self.assertIn("uses extracted frames", result["error"])


if __name__ == "__main__":
    unittest.main()
