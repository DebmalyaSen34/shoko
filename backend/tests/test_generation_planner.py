import os
import json
import shutil
import tempfile
import unittest
from unittest import mock

from src.workflows.generation_planner import plan_generation_workflow

class TestGenerationPlannerWorkflow(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.feedback_path = os.path.join(self.test_dir, "feedback.json")
        self.output_base_dir = os.path.join(self.test_dir, "data")

        # Mock feedback.json content (Option A grouped structure)
        self.mock_feedback = [
            {
                "clip_used": "clip3.mp4",
                "previous_clip": "clip2.mp4",
                "clip_start_tc": "00:00:20:00",
                "clip_end_tc": "00:00:30:00",
                "clip_start_s": 20.0,
                "clip_end_s": 30.0,
                "audio_used": None,
                "feedback_items": [
                    {
                        "timestamp": "00:24",
                        "category": "video",
                        "remark": "Zoom in reaction"
                    }
                ]
            },
            {
                "clip_used": "clip1.mp4",
                "previous_clip": None,
                "clip_start_tc": "00:00:00:00",
                "clip_end_tc": "00:00:10:00",
                "clip_start_s": 0.0,
                "clip_end_s": 10.0,
                "audio_used": "audio1.mp3",
                "feedback_items": [
                    {
                        "timestamp": "00:02",
                        "category": "audio",
                        "remark": "Dubbing adjustment"
                    }
                ]
            },
            {
                "clip_used": "clip2.mp4",
                "previous_clip": "clip1.mp4",
                "clip_start_tc": "00:00:10:00",
                "clip_end_tc": "00:00:20:00",
                "clip_start_s": 10.0,
                "clip_end_s": 20.0,
                "audio_used": None,
                "feedback_items": [
                    {
                        "timestamp": "00:12",
                        "category": "video",
                        "remark": "Vir should look playful"
                    }
                ]
            }
        ]

        with open(self.feedback_path, "w", encoding="utf-8") as f:
            json.dump(self.mock_feedback, f)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @mock.patch("src.workflows.generation_planner.generate_structured")
    def test_plan_generation_workflow_successful(self, mock_gen_structured):
        # Arrange
        # Mock LLM planning output
        mock_gen_structured.return_value = {
            "plans": [
                {
                    "clip_used": "clip3.mp4",
                    "previous_clip": "clip2.mp4",
                    "clip_start_tc": "00:00:20:00",
                    "clip_end_tc": "00:00:30:00",
                    "clip_start_s": 20.0,
                    "clip_end_s": 30.0,
                    "generation_type": "simple",
                    "classification_reasoning": "Zoom in reaction is a post-processing camera move.",
                    "audio_used": None,
                    "is_dialogue_active": False,
                    "characters_present": ["Rian"],
                    "location": "hall",
                    "requires_previous_clip_continuity": False,
                    "remarks_to_process": ["Zoom in reaction"]
                },
                {
                    "clip_used": "clip1.mp4",
                    "previous_clip": None,
                    "clip_start_tc": "00:00:00:00",
                    "clip_end_tc": "00:00:10:00",
                    "clip_start_s": 0.0,
                    "clip_end_s": 10.0,
                    "generation_type": "none",
                    "classification_reasoning": "Dubbing adjustment is audio only.",
                    "audio_used": "audio1.mp3",
                    "is_dialogue_active": True,
                    "characters_present": [],
                    "location": None,
                    "requires_previous_clip_continuity": False,
                    "remarks_to_process": []
                },
                {
                    "clip_used": "clip2.mp4",
                    "previous_clip": "clip1.mp4",
                    "clip_start_tc": "00:00:10:00",
                    "clip_end_tc": "00:00:20:00",
                    "clip_start_s": 10.0,
                    "clip_end_s": 20.0,
                    "generation_type": "complex",
                    "classification_reasoning": "Action change (playful movement) requires character reference sheets.",
                    "audio_used": None,
                    "is_dialogue_active": False,
                    "characters_present": ["Vir"],
                    "location": "hall",
                    "requires_previous_clip_continuity": True,
                    "previous_clip_dependency_reason": "Vir entering hall matches previous background setting.",
                    "remarks_to_process": ["Vir should look playful"]
                }
            ]
        }

        project_name = "test_plan_project"
        mock_client = mock.MagicMock()

        # Mock os.path.exists to simulate that audio1.mp3 exists locally
        orig_exists = os.path.exists
        def mock_exists(path):
            if "audio1.mp3" in path:
                return True
            return orig_exists(path)

        # Act
        with mock.patch("os.path.exists", side_effect=mock_exists):
            result_path = plan_generation_workflow(
                feedback_json_path=self.feedback_path,
                project_name=project_name,
                openai_client=mock_client,
                output_base_dir=self.output_base_dir
            )

        # Assert
        expected_output_path = os.path.abspath(
            os.path.join(self.output_base_dir, project_name, "generation_plan.json")
        )
        self.assertEqual(result_path, expected_output_path)
        self.assertTrue(os.path.exists(result_path))

        with open(result_path, "r", encoding="utf-8") as f:
            plan_data = json.load(f)

        # Verify chronological sorting (0.0s, then 10.0s, then 20.0s)
        self.assertEqual(len(plan_data), 3)
        self.assertEqual(plan_data[0]["clip_used"], "clip1.mp4")
        self.assertEqual(plan_data[0]["clip_start_s"], 0.0)
        self.assertEqual(plan_data[0]["generation_type"], "none")
        self.assertEqual(plan_data[0]["audio_used"], "audio1.mp3")
        self.assertTrue(plan_data[0]["is_dialogue_active"])
        # Verify audio_path resolution
        self.assertIsNotNone(plan_data[0]["audio_path"])
        self.assertTrue(plan_data[0]["audio_path"].endswith("audio1.mp3"))

        self.assertEqual(plan_data[1]["clip_used"], "clip2.mp4")
        self.assertEqual(plan_data[1]["clip_start_s"], 10.0)
        self.assertEqual(plan_data[1]["generation_type"], "complex")
        self.assertEqual(plan_data[1]["characters_present"], ["Vir"])
        self.assertTrue(plan_data[1]["requires_previous_clip_continuity"])

        self.assertEqual(plan_data[2]["clip_used"], "clip3.mp4")
        self.assertEqual(plan_data[2]["clip_start_s"], 20.0)
        self.assertEqual(plan_data[2]["generation_type"], "simple")

    def test_missing_feedback_file(self):
        # Arrange
        non_existent_path = os.path.join(self.test_dir, "missing.json")

        # Act & Assert
        with self.assertRaises(FileNotFoundError):
            plan_generation_workflow(
                feedback_json_path=non_existent_path,
                project_name="my_project",
                openai_client=mock.MagicMock(),
                output_base_dir=self.output_base_dir
            )

    @mock.patch("src.workflows.generation_planner.generate_structured")
    def test_absent_requested_subject_forces_complex_compound_plan(self, mock_gen_structured):
        mock_feedback = [
            {
                "clip_used": "output (23).mp4",
                "clip_occurrence": 7,
                "previous_clip": "previous.mp4",
                "clip_start_tc": "00:00:44:00",
                "clip_end_tc": "00:00:48:00",
                "clip_start_s": 44.0,
                "clip_end_s": 48.0,
                "audio_used": None,
                "feedback_items": [
                    {
                        "timestamp": "00:45",
                        "category": "video",
                        "remark": "Show Vir grasping his mother's hand and acting innocent."
                    },
                    {
                        "timestamp": "00:46",
                        "category": "video",
                        "remark": "Have mother begin to get angry and look down."
                    }
                ]
            }
        ]
        with open(self.feedback_path, "w", encoding="utf-8") as file:
            json.dump(mock_feedback, file)

        mock_gen_structured.return_value = {
            "plans": [
                {
                    "clip_used": "output (23).mp4",
                    "previous_clip": "previous.mp4",
                    "clip_start_tc": "00:00:44:00",
                    "clip_end_tc": "00:00:48:00",
                    "clip_start_s": 44.0,
                    "clip_end_s": 48.0,
                    "generation_type": "simple",
                    "classification_reasoning": "Mother expression edit.",
                    "audio_used": None,
                    "is_dialogue_active": False,
                    "characters_present": ["Mother", "Vir"],
                    "location": None,
                    "requires_previous_clip_continuity": False,
                    "remarks_to_process": ["Show Vir grasping his mother's hand and acting innocent."],
                    "absent_requested_subjects": []
                }
            ]
        }

        result_path = plan_generation_workflow(
            feedback_json_path=self.feedback_path,
            project_name="test_plan_project",
            openai_client=mock.MagicMock(),
            output_base_dir=self.output_base_dir,
        )

        with open(result_path, "r", encoding="utf-8") as file:
            plan_data = json.load(file)

        self.assertEqual(1, len(plan_data))
        self.assertEqual("complex", plan_data[0]["generation_type"])
        self.assertEqual(["Vir"], plan_data[0]["absent_requested_subjects"])
        self.assertIn("Vir", plan_data[0]["characters_present"])
        self.assertTrue(plan_data[0]["compound_feedback"])
        self.assertEqual(["00:45", "00:46"], plan_data[0]["source_feedback_timestamps"])
        self.assertEqual(
            [
                "Show Vir grasping his mother's hand and acting innocent.",
                "Have mother begin to get angry and look down.",
            ],
            plan_data[0]["remarks_to_process"],
        )

if __name__ == "__main__":
    unittest.main()
