import os
import json
import shutil
import tempfile
import unittest
from unittest import mock

from src.workflows.prompt_generation import extract_frames_per_second, generate_video_prompts_from_plan

class TestPromptGenerationWorkflow(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.plan_path = os.path.join(self.test_dir, "generation_plan.json")
        self.assets_dir = os.path.join(self.test_dir, "assets")
        self.project_assets_dir = os.path.join(self.assets_dir, "test_prompt_project")
        self.output_base_dir = os.path.join(self.test_dir, "data")

        # Create assets subdirectories for testing under project subfolder
        os.makedirs(os.path.join(self.project_assets_dir, "01_characters"), exist_ok=True)
        os.makedirs(os.path.join(self.project_assets_dir, "03_locations"), exist_ok=True)
        os.makedirs(os.path.join(self.project_assets_dir, "06_clips", "_raw"), exist_ok=True)
        os.makedirs(os.path.join(self.project_assets_dir, "04_audio"), exist_ok=True)

        # Create dummy character and location sheets
        self.vir_sheet = os.path.join(self.project_assets_dir, "01_characters", "vir-sheet-v2.png")
        with open(self.vir_sheet, "w") as f:
            f.write("character sheet vir")

        self.hall_sheet = os.path.join(self.project_assets_dir, "03_locations", "hall-v1.png")
        with open(self.hall_sheet, "w") as f:
            f.write("location sheet hall")

        # Create dummy clips
        self.clip1_path = os.path.join(self.project_assets_dir, "06_clips", "_raw", "clip1.mp4")
        with open(self.clip1_path, "w") as f:
            f.write("clip1 video")

        self.clip2_path = os.path.join(self.project_assets_dir, "06_clips", "_raw", "clip2.mp4")
        with open(self.clip2_path, "w") as f:
            f.write("clip2 video")

        # Create dummy audio
        self.audio1_path = os.path.join(self.project_assets_dir, "04_audio", "audio1.mp3")
        with open(self.audio1_path, "w") as f:
            f.write("audio1 data")

        # Create a mock generation_plan.json
        self.mock_plan = [
            {
                "clip_used": "clip1.mp4",
                "previous_clip": None,
                "clip_start_tc": "00:00:00:00",
                "clip_end_tc": "00:00:10:00",
                "clip_start_s": 0.0,
                "clip_end_s": 10.0,
                "generation_type": "none",
                "classification_reasoning": "Audio feedback only.",
                "audio_used": "audio1.mp3",
                "audio_path": self.audio1_path,
                "is_dialogue_active": False,
                "characters_present": [],
                "location": None,
                "requires_previous_clip_continuity": False,
                "remarks_to_process": []
            },
            {
                "clip_used": "clip2.mp4",
                "previous_clip": "clip1.mp4",
                "clip_start_tc": "00:00:10:00",
                "clip_end_tc": "00:00:15:00",
                "clip_start_s": 10.0,
                "clip_end_s": 15.0,
                "generation_type": "complex",
                "classification_reasoning": "Vir performance change.",
                "audio_used": "audio1.mp3",
                "audio_path": self.audio1_path,
                "is_dialogue_active": True,
                "characters_present": ["Vir"],
                "location": "hall",
                "requires_previous_clip_continuity": True,
                "remarks_to_process": ["Vir should enter hall playfully"]
            }
        ]

        with open(self.plan_path, "w", encoding="utf-8") as f:
            json.dump(self.mock_plan, f)

        # Create dummy trimmed audio file under the expected directory for clip2
        clip_frames_dir = os.path.join(self.output_base_dir, "test_prompt_project", "clip_frames", "clip2")
        os.makedirs(clip_frames_dir, exist_ok=True)
        trimmed_audio_path = os.path.join(clip_frames_dir, "trimmed_audio1.mp3")
        with open(trimmed_audio_path, "w") as f:
            f.write("dummy audio content")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @mock.patch("src.workflows.prompt_generation.os.path.exists")
    @mock.patch("src.workflows.prompt_generation.subprocess.run")
    def test_extract_frames_per_second_uses_adaptive_full_clip_offsets(
        self, mock_run, mock_exists
    ):
        mock_run.return_value = mock.Mock(returncode=0, stderr="")
        mock_exists.return_value = True

        frame_paths = extract_frames_per_second(
            video_path=self.clip2_path,
            output_dir=os.path.join(self.test_dir, "adaptive_frames"),
            duration_s=6.0,
        )

        self.assertEqual(7, len(frame_paths))
        self.assertTrue(frame_paths[0].endswith("frame_001.jpg"))
        self.assertTrue(frame_paths[-1].endswith("frame_007.jpg"))
        offsets = [call.args[0][3] for call in mock_run.call_args_list]
        self.assertEqual("0.050", offsets[0])
        self.assertEqual("5.900", offsets[-1])

    @mock.patch("src.workflows.prompt_generation.generate_structured")
    @mock.patch("src.workflows.prompt_generation._file_data_url")
    @mock.patch("src.workflows.prompt_generation.extract_frames_per_second")
    @mock.patch("src.workflows.prompt_generation.extract_last_frame")
    @mock.patch("src.workflows.prompt_generation.extract_audio_segment")
    @mock.patch("src.workflows.prompt_generation.get_video_duration")
    @mock.patch("src.workflows.prompt_generation.analyze_clip_context")
    def test_generate_video_prompts_from_plan_successful(
        self, mock_clip_context, mock_get_duration, mock_extract_audio, mock_last_frame, mock_frames_fps, mock_file_url, mock_gen_structured
    ):
        # Arrange
        mock_clip_context.return_value = {}
        mock_get_duration.return_value = 5.0
        mock_extract_audio.return_value = True
        mock_file_url.return_value = "data:image/png;base64,dummy_data"
        mock_frames_fps.return_value = ["/dummy/frame1.jpg"]
        mock_last_frame.return_value = "/dummy/last_frame.jpg"

        from src.workflows.prompt_generation import DialogueAssessmentResult, ContinuityVerificationResult
        def mock_gen_side_effect(*args, **kwargs):
            schema = kwargs.get("schema")
            if schema == DialogueAssessmentResult:
                return {
                    "is_dialogue_active": True,
                    "reasoning": "Spoken dialogue detected in test frames."
                }
            elif schema == ContinuityVerificationResult:
                return {
                    "requires_previous_clip_continuity": True,
                    "reasoning": "Continuity matches in mock."
                }
            else:
                return {
                    "video_model_prompt": "Warm sunlit hall... Vir enters playfully... mouth moving in sync.",
                    "explanation": "Addressed Vir playful entry and resolved dialogue sync."
                }
        mock_gen_structured.side_effect = mock_gen_side_effect

        project_name = "test_prompt_project"
        mock_client = mock.MagicMock()
        mock_transcript = mock.MagicMock()
        mock_transcript.text = "Kya hua?"
        mock_client.audio.transcriptions.create.return_value = mock_transcript

        # Act
        result_path = generate_video_prompts_from_plan(
            plan_json_path=self.plan_path,
            project_name=project_name,
            assets_dir=self.assets_dir,
            openai_client=mock_client,
            output_base_dir=self.output_base_dir
        )

        # Assert
        expected_output_path = os.path.abspath(
            os.path.join(self.output_base_dir, project_name, "video_prompts.json")
        )
        self.assertEqual(result_path, expected_output_path)
        self.assertTrue(os.path.exists(result_path))

        with open(result_path, "r", encoding="utf-8") as f:
            prompts_data = json.load(f)

        self.assertEqual(len(prompts_data), 2)

        # Clip 1 (none)
        self.assertEqual(prompts_data[0]["clip_used"], "clip1.mp4")
        self.assertEqual(prompts_data[0]["generation_type"], "none")
        self.assertEqual(prompts_data[0]["video_model_prompt"], "")

        # Clip 2 (complex)
        self.assertEqual(prompts_data[1]["clip_used"], "clip2.mp4")
        self.assertEqual(prompts_data[1]["generation_type"], "complex")
        self.assertEqual(prompts_data[1]["category"], "video")
        self.assertIn("REFERENCE IMAGE MAP:", prompts_data[1]["video_model_prompt"])
        self.assertTrue(
            prompts_data[1]["video_model_prompt"].endswith(
                "Warm sunlit hall... Vir enters playfully... mouth moving in sync."
            )
        )
        # Verify resolved assets
        self.assertIn(self.vir_sheet, prompts_data[1]["selected_assets"])
        self.assertIn(self.hall_sheet, prompts_data[1]["selected_assets"])
        # Verify audio reference setup
        self.assertTrue(prompts_data[1]["is_dialogue_active"])
        self.assertFalse(prompts_data[1]["generate_audio"])
        self.assertTrue(prompts_data[1]["has_reference_audio"])
        self.assertEqual(prompts_data[1]["audio_url"], "data:image/png;base64,dummy_data")
        self.assertEqual(prompts_data[1]["audio_trim_start_s"], 10.0)
        self.assertEqual(prompts_data[1]["audio_trim_end_s"], 15.0)
        self.assertEqual(prompts_data[1]["audio_trim_source"], "sequence_timeline")
        # Verify first_frame continuity setup
        self.assertEqual(prompts_data[1]["first_frame_url"], "data:image/png;base64,dummy_data")
        self.assertIsNotNone(prompts_data[1].get("initial_frame_image_path"))
        # Verify clip frames paths are populated
        self.assertEqual(prompts_data[1]["clip_frame_paths"], ["/dummy/frame1.jpg"])
        # Verify duration clamping (5s duration)
        self.assertEqual(prompts_data[1]["duration"], 5)

    @mock.patch("src.workflows.prompt_generation.generate_structured")
    @mock.patch("src.workflows.prompt_generation._file_data_url")
    @mock.patch("src.workflows.prompt_generation.extract_frames_per_second")
    @mock.patch("src.workflows.prompt_generation.extract_last_frame")
    @mock.patch("src.workflows.prompt_generation.extract_audio_segment")
    @mock.patch("src.workflows.prompt_generation.get_video_duration")
    @mock.patch("src.workflows.prompt_generation.analyze_clip_context")
    def test_generate_video_prompts_from_plan_with_transcription(
        self, mock_clip_context, mock_get_duration, mock_extract_audio, mock_last_frame, mock_frames_fps, mock_file_url, mock_gen_structured
    ):
        # Arrange
        mock_clip_context.return_value = {}
        mock_get_duration.return_value = 5.0
        mock_extract_audio.return_value = True
        mock_file_url.return_value = "data:audio/mp3;base64,dummy_audio"
        mock_frames_fps.return_value = ["/dummy/frame1.jpg"]
        mock_last_frame.return_value = "/dummy/last_frame.jpg"

        project_name = "test_prompt_project"
        mock_client = mock.MagicMock()
        mock_transcript = mock.MagicMock()
        mock_transcript.text = "Kya hua?"
        mock_client.audio.transcriptions.create.return_value = mock_transcript

        from src.workflows.prompt_generation import DialogueAssessmentResult, ContinuityVerificationResult
        def mock_gen_side_effect(*args, **kwargs):
            schema = kwargs.get("schema")
            if schema == DialogueAssessmentResult:
                return {
                    "is_dialogue_active": False,
                    "reasoning": "Should not be called"
                }
            elif schema == ContinuityVerificationResult:
                return {
                    "requires_previous_clip_continuity": False,
                    "reasoning": "Continuity matches in mock."
                }
            else:
                contents = kwargs.get("contents", [])
                instruction_text = contents[-1] if contents else ""
                self.assertIn("trimmed reference audio segment from timeline 10.000s to 15.000s", instruction_text)
                self.assertIn("Do not quote or invent transcript text", instruction_text)
                return {
                    "video_model_prompt": "Warm sunlit hall... Mother shouts 'Kya hua?'... mouth moving in sync.",
                    "explanation": "Addressed mother shouting."
                }
        mock_gen_structured.side_effect = mock_gen_side_effect

        # Act
        result_path = generate_video_prompts_from_plan(
            plan_json_path=self.plan_path,
            project_name=project_name,
            assets_dir=self.assets_dir,
            openai_client=mock_client,
            output_base_dir=self.output_base_dir
        )

        # Assert
        with open(result_path, "r", encoding="utf-8") as f:
            prompts_data = json.load(f)

        self.assertTrue(prompts_data[1]["is_dialogue_active"])
        self.assertFalse(prompts_data[1]["generate_audio"])
        self.assertTrue(prompts_data[1]["has_reference_audio"])
        self.assertEqual(prompts_data[1]["audio_url"], "data:audio/mp3;base64,dummy_audio")
        self.assertIn("REFERENCE IMAGE MAP:", prompts_data[1]["video_model_prompt"])
        self.assertTrue(
            prompts_data[1]["video_model_prompt"].endswith(
                "Warm sunlit hall... Mother shouts 'Kya hua?'... mouth moving in sync."
            )
        )

    @mock.patch("src.workflows.prompt_generation.generate_structured")
    @mock.patch("src.workflows.prompt_generation._file_data_url")
    @mock.patch("src.workflows.prompt_generation.extract_frames_per_second")
    @mock.patch("src.workflows.prompt_generation.extract_last_frame")
    @mock.patch("src.workflows.prompt_generation.extract_audio_segment")
    @mock.patch("src.workflows.prompt_generation.get_video_duration")
    @mock.patch("src.workflows.prompt_generation.analyze_clip_context")
    def test_generate_video_prompts_from_plan_preserves_referenced_frames(
        self, mock_clip_context, mock_get_duration, mock_extract_audio, mock_last_frame, mock_frames_fps, mock_file_url, mock_gen_structured
    ):
        mock_clip_context.return_value = {}
        ref_frame_path = os.path.join(self.test_dir, "ref_00_44.jpg")
        with open(ref_frame_path, "wb") as file:
            file.write(b"referenced frame")
        self.mock_plan[1]["referenced_frames"] = [
            {
                "timestamp": "00:44",
                "reason": "Vir evil smile cutaway",
                "frame_path": ref_frame_path,
                "clip_used": "smile.mp4",
                "offset_s": 0.0,
            },
            {
                "timestamp": "00:44",
                "reason": "Vir evil smile cutaway duplicate",
                "frame_path": ref_frame_path,
                "clip_used": "smile.mp4",
                "offset_s": 0.0,
            }
        ]
        with open(self.plan_path, "w", encoding="utf-8") as file:
            json.dump(self.mock_plan, file)

        mock_get_duration.return_value = 5.0
        mock_extract_audio.return_value = False
        mock_file_url.return_value = "data:image/jpeg;base64,dummy_data"
        mock_frames_fps.return_value = ["/dummy/frame1.jpg"]
        mock_last_frame.return_value = "/dummy/last_frame.jpg"

        def mock_gen_side_effect(*args, **kwargs):
            instruction_text = kwargs.get("contents", [])[-1]
            self.assertIn("@ref1", instruction_text)
            self.assertIn("Vir evil smile cutaway", instruction_text)
            return {
                "video_model_prompt": "Mother reacts, then hard-cuts to @ref1 for Vir's smile.",
                "explanation": "Added the referenced cutaway frame.",
            }
        mock_gen_structured.side_effect = mock_gen_side_effect

        result_path = generate_video_prompts_from_plan(
            plan_json_path=self.plan_path,
            project_name="test_prompt_project",
            assets_dir=self.assets_dir,
            openai_client=mock.MagicMock(),
            output_base_dir=self.output_base_dir,
        )

        with open(result_path, "r", encoding="utf-8") as file:
            prompts_data = json.load(file)

        self.assertEqual([self.mock_plan[1]["referenced_frames"][0]], prompts_data[1]["referenced_frames"])
        self.assertEqual([ref_frame_path], prompts_data[1]["referenced_frame_paths"])
        self.assertEqual(["@ref1"], prompts_data[1]["referenced_frame_labels"])
        self.assertIn("image 4", prompts_data[1]["video_model_prompt"])
        self.assertNotIn("@ref1", prompts_data[1]["video_model_prompt"])
        self.assertNotIn("first_frame_url", prompts_data[1])
        self.assertNotIn("initial_frame_image_path", prompts_data[1])

    @mock.patch("src.workflows.prompt_generation.generate_structured")
    @mock.patch("src.workflows.prompt_generation._file_data_url")
    @mock.patch("src.workflows.prompt_generation.extract_frames_per_second")
    @mock.patch("src.workflows.prompt_generation.extract_last_frame")
    @mock.patch("src.workflows.prompt_generation.extract_audio_segment")
    @mock.patch("src.workflows.prompt_generation.get_video_duration")
    @mock.patch("src.workflows.prompt_generation.analyze_clip_context")
    def test_generate_video_prompts_from_plan_short_duration_and_empty_transcript(
        self, mock_clip_context, mock_get_duration, mock_extract_audio, mock_last_frame, mock_frames_fps, mock_file_url, mock_gen_structured
    ):
        # Arrange: local plan with a short duration clip (1.5s < 1.8s) and a clip with empty transcript
        mock_clip_context.return_value = {}
        local_plan = [
            {
                "clip_used": "clip2.mp4",
                "previous_clip": "clip1.mp4",
                "clip_start_tc": "00:00:10:00",
                "clip_end_tc": "00:00:11:12",
                "clip_start_s": 10.0,
                "clip_end_s": 11.5,  # 1.5s duration
                "generation_type": "complex",
                "classification_reasoning": "Vir performance change.",
                "audio_used": "audio1.mp3",
                "audio_path": self.audio1_path,
                "is_dialogue_active": True,
                "characters_present": ["Vir"],
                "location": "hall",
                "requires_previous_clip_continuity": True,
                "remarks_to_process": ["Vir should enter hall playfully"]
            },
            {
                "clip_used": "clip3.mp4",
                "previous_clip": "clip2.mp4",
                "clip_start_tc": "00:00:11:12",
                "clip_end_tc": "00:00:16:12",
                "clip_start_s": 11.5,
                "clip_end_s": 16.5,  # 5.0s duration
                "generation_type": "complex",
                "classification_reasoning": "Vir performance change.",
                "audio_used": "audio1.mp3",
                "audio_path": self.audio1_path,
                "is_dialogue_active": True,
                "characters_present": ["Vir"],
                "location": "hall",
                "requires_previous_clip_continuity": True,
                "remarks_to_process": ["Vir should enter hall playfully"]
            }
        ]

        local_plan_path = os.path.join(self.test_dir, "local_plan.json")
        with open(local_plan_path, "w", encoding="utf-8") as f:
            json.dump(local_plan, f)

        # Create dummy files
        for clip in ["clip2", "clip3"]:
            clip_frames_dir = os.path.join(self.output_base_dir, "test_prompt_project", "clip_frames", clip)
            os.makedirs(clip_frames_dir, exist_ok=True)
            with open(os.path.join(clip_frames_dir, "trimmed_audio1.mp3"), "w") as f:
                f.write("dummy audio content")

        mock_get_duration.return_value = 5.0
        mock_extract_audio.return_value = True
        mock_file_url.return_value = "data:audio/mp3;base64,dummy_audio"
        mock_frames_fps.return_value = ["/dummy/frame1.jpg"]
        mock_last_frame.return_value = "/dummy/last_frame.jpg"

        mock_client = mock.MagicMock()
        mock_transcript = mock.MagicMock()
        mock_transcript.text = ""  # Empty transcription
        mock_client.audio.transcriptions.create.return_value = mock_transcript

        from src.workflows.prompt_generation import DialogueAssessmentResult, ContinuityVerificationResult
        def mock_gen_side_effect(*args, **kwargs):
            return {
                "video_model_prompt": "Warm sunlit hall... Mother is silent.",
                "explanation": "No dialogue sync."
            }
        mock_gen_structured.side_effect = mock_gen_side_effect

        # Act
        result_path = generate_video_prompts_from_plan(
            plan_json_path=local_plan_path,
            project_name="test_prompt_project",
            assets_dir=self.assets_dir,
            openai_client=mock_client,
            output_base_dir=self.output_base_dir
        )

        # Assert
        with open(result_path, "r", encoding="utf-8") as f:
            prompts_data = json.load(f)

        # Both clips keep trimmed reference audio even when short or untranscribed.
        # clip2 (duration < 1.8s)
        self.assertTrue(prompts_data[0]["is_dialogue_active"])
        self.assertFalse(prompts_data[0]["generate_audio"])
        self.assertTrue(prompts_data[0]["has_reference_audio"])
        self.assertEqual(prompts_data[0]["audio_url"], "data:audio/mp3;base64,dummy_audio")
        self.assertEqual(prompts_data[0]["audio_trim_start_s"], 10.0)
        self.assertEqual(prompts_data[0]["audio_trim_end_s"], 11.5)

        # clip3 (duration >= 1.8s and no transcript requirement)
        self.assertTrue(prompts_data[1]["is_dialogue_active"])
        self.assertFalse(prompts_data[1]["generate_audio"])
        self.assertTrue(prompts_data[1]["has_reference_audio"])
        self.assertEqual(prompts_data[1]["audio_url"], "data:audio/mp3;base64,dummy_audio")
        self.assertEqual(prompts_data[1]["audio_trim_start_s"], 11.5)
        self.assertEqual(prompts_data[1]["audio_trim_end_s"], 16.5)

    def test_missing_plan_file(self):
        # Arrange
        non_existent_path = os.path.join(self.test_dir, "missing.json")

        # Act & Assert
        with self.assertRaises(FileNotFoundError):
            generate_video_prompts_from_plan(
                plan_json_path=non_existent_path,
                project_name="my_project",
                assets_dir=self.assets_dir,
                openai_client=mock.MagicMock(),
                output_base_dir=self.output_base_dir
            )

    @mock.patch("src.workflows.prompt_generation.analyze_clip_context")
    @mock.patch("src.workflows.prompt_generation.generate_structured")
    @mock.patch("src.workflows.prompt_generation._file_data_url")
    @mock.patch("src.workflows.prompt_generation.extract_frames_per_second")
    def test_generate_video_prompts_includes_saved_clip_context(
        self, mock_frames_fps, mock_file_url, mock_gen_structured, mock_clip_context
    ):
        local_plan = [
            {
                "clip_used": "clip2.mp4",
                "previous_clip": None,
                "clip_start_tc": "00:00:10:00",
                "clip_end_tc": "00:00:15:00",
                "clip_start_s": 10.0,
                "clip_end_s": 15.0,
                "generation_type": "simple",
                "classification_reasoning": "Mother expression edit.",
                "audio_used": None,
                "audio_path": None,
                "is_dialogue_active": False,
                "characters_present": ["Mother"],
                "location": None,
                "requires_previous_clip_continuity": False,
                "remarks_to_process": ["Have mother begin to get angry and look down."],
                "source_feedback_timestamps": ["00:46"],
            }
        ]
        local_plan_path = os.path.join(self.test_dir, "context_plan.json")
        with open(local_plan_path, "w", encoding="utf-8") as file:
            json.dump(local_plan, file)

        mock_clip_context.return_value = {
            "summary": "Mother is looking into the camera without emotion.",
            "visible_characters": ["Mother"],
            "expressions": ["neutral"],
            "gaze": ["looking into camera"],
            "actions": [],
            "blocking": ["Mother in foreground"],
            "camera_framing": "close shot",
            "location": "interior",
            "continuity_notes": [],
            "uncertainty_flags": [],
            "status": "video_and_frames",
            "clip_context_path": os.path.join(self.output_base_dir, "context.json"),
            "clip_segment_path": os.path.join(self.output_base_dir, "segment.mp4"),
            "frame_paths": ["/dummy/context_frame.jpg"],
        }
        mock_file_url.return_value = "data:image/jpeg;base64,frame"
        mock_frames_fps.return_value = ["/dummy/fallback_frame.jpg"]

        def mock_gen_side_effect(*args, **kwargs):
            instruction_text = kwargs.get("contents", [])[-1]
            self.assertIn("Saved Clip Understanding Context", instruction_text)
            self.assertIn("Mother is looking into the camera without emotion", instruction_text)
            self.assertLess(
                instruction_text.index("Saved Clip Understanding Context"),
                instruction_text.index("Client Feedback Remarks"),
            )
            return {
                "video_model_prompt": "Mother shifts from neutral camera gaze to anger and looks down.",
                "explanation": "Used saved context.",
            }
        mock_gen_structured.side_effect = mock_gen_side_effect

        result_path = generate_video_prompts_from_plan(
            plan_json_path=local_plan_path,
            project_name="test_prompt_project",
            assets_dir=self.assets_dir,
            openai_client=mock.MagicMock(),
            output_base_dir=self.output_base_dir,
        )

        with open(result_path, "r", encoding="utf-8") as file:
            prompts_data = json.load(file)

        self.assertEqual("video_and_frames", prompts_data[0]["clip_context_status"])
        self.assertTrue(prompts_data[0]["clip_segment_path"].endswith("segment.mp4"))

    @mock.patch("src.workflows.prompt_generation.generate_structured")
    @mock.patch("src.workflows.prompt_generation._file_data_url")
    @mock.patch("src.workflows.prompt_generation.extract_frames_per_second")
    @mock.patch("src.workflows.prompt_generation.extract_audio_segment")
    @mock.patch("src.workflows.prompt_generation.analyze_clip_context")
    def test_absent_requested_character_uses_character_sheet(
        self, mock_clip_context, mock_extract_audio, mock_frames_fps, mock_file_url, mock_gen_structured
    ):
        mock_clip_context.return_value = {}
        local_plan = [
            {
                "clip_used": "clip2.mp4",
                "previous_clip": None,
                "clip_start_tc": "00:00:10:00",
                "clip_end_tc": "00:00:15:00",
                "clip_start_s": 10.0,
                "clip_end_s": 15.0,
                "generation_type": "complex",
                "classification_reasoning": "Vir must be added to the shot.",
                "audio_used": None,
                "audio_path": None,
                "is_dialogue_active": False,
                "characters_present": ["Mother"],
                "location": None,
                "requires_previous_clip_continuity": False,
                "remarks_to_process": ["Show Vir grasping his mother's hand."],
                "compound_feedback": True,
                "source_feedback_timestamps": ["00:45", "00:46"],
                "absent_requested_subjects": ["Vir"],
            }
        ]
        local_plan_path = os.path.join(self.test_dir, "absent_plan.json")
        with open(local_plan_path, "w", encoding="utf-8") as file:
            json.dump(local_plan, file)

        mock_extract_audio.return_value = False
        mock_file_url.return_value = "data:image/png;base64,dummy_data"
        mock_frames_fps.return_value = ["/dummy/frame1.jpg"]

        def mock_gen_side_effect(*args, **kwargs):
            instruction_text = kwargs.get("contents", [])[-1]
            self.assertIn("explicitly requested by feedback but are not present", instruction_text)
            self.assertIn("Vir", instruction_text)
            self.assertIn("compound same-clip edit", instruction_text)
            return {
                "video_model_prompt": "Vir enters the shot using image 2 identity, grasps his mother's hand.",
                "explanation": "Used Vir sheet for the added subject.",
            }
        mock_gen_structured.side_effect = mock_gen_side_effect

        result_path = generate_video_prompts_from_plan(
            plan_json_path=local_plan_path,
            project_name="test_prompt_project",
            assets_dir=self.assets_dir,
            openai_client=mock.MagicMock(),
            output_base_dir=self.output_base_dir,
        )

        with open(result_path, "r", encoding="utf-8") as file:
            prompts_data = json.load(file)

        self.assertEqual(["Vir"], prompts_data[0]["absent_requested_subjects"])
        self.assertEqual([], prompts_data[0]["missing_required_subject_sheets"])
        self.assertIn(self.vir_sheet, prompts_data[0]["selected_assets"])
        self.assertFalse(prompts_data[0]["prompt_generation_review_required"])
        self.assertEqual("success", prompts_data[0]["status"])

    @mock.patch("src.workflows.prompt_generation.generate_structured")
    @mock.patch("src.workflows.prompt_generation._file_data_url")
    @mock.patch("src.workflows.prompt_generation.extract_frames_per_second")
    @mock.patch("src.workflows.prompt_generation.extract_audio_segment")
    @mock.patch("src.workflows.prompt_generation.analyze_clip_context")
    def test_missing_absent_character_sheet_marks_review_needed(
        self, mock_clip_context, mock_extract_audio, mock_frames_fps, mock_file_url, mock_gen_structured
    ):
        mock_clip_context.return_value = {}
        os.remove(self.vir_sheet)
        local_plan = [
            {
                "clip_used": "clip2.mp4",
                "previous_clip": None,
                "clip_start_tc": "00:00:10:00",
                "clip_end_tc": "00:00:15:00",
                "clip_start_s": 10.0,
                "clip_end_s": 15.0,
                "generation_type": "complex",
                "classification_reasoning": "Vir must be added to the shot.",
                "audio_used": None,
                "audio_path": None,
                "is_dialogue_active": False,
                "characters_present": ["Vir"],
                "location": None,
                "requires_previous_clip_continuity": False,
                "remarks_to_process": ["Show Vir grasping his mother's hand."],
                "absent_requested_subjects": ["Vir"],
            }
        ]
        local_plan_path = os.path.join(self.test_dir, "missing_sheet_plan.json")
        with open(local_plan_path, "w", encoding="utf-8") as file:
            json.dump(local_plan, file)

        mock_extract_audio.return_value = False
        mock_file_url.return_value = "data:image/png;base64,dummy_data"
        mock_frames_fps.return_value = ["/dummy/frame1.jpg"]

        result_path = generate_video_prompts_from_plan(
            plan_json_path=local_plan_path,
            project_name="test_prompt_project",
            assets_dir=self.assets_dir,
            openai_client=mock.MagicMock(),
            output_base_dir=self.output_base_dir,
        )

        with open(result_path, "r", encoding="utf-8") as file:
            prompts_data = json.load(file)

        mock_gen_structured.assert_not_called()
        self.assertEqual("warning", prompts_data[0]["status"])
        self.assertTrue(prompts_data[0]["prompt_generation_review_required"])
        self.assertEqual(["Vir"], prompts_data[0]["missing_required_subject_sheets"])

if __name__ == "__main__":
    unittest.main()
