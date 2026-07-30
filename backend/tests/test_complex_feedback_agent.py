import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

from src.workflows.complex_feedback_agent import (
    ComplexFeedbackInvestigationResult,
    ComplexFeedbackPlan,
    ComplexReferencePlan,
    build_deterministic_complex_feedback_plans,
    investigate_complex_feedback_references,
    segment_needs_complex_reference_investigation,
)


class TestComplexFeedbackAgentWorkflow(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.project_name = "project-a"
        self.feedback_path = os.path.join(self.test_dir, "feedback.json")
        self.timeline_path = os.path.join(self.test_dir, "timeline.json")
        self.assets_dir = os.path.join(self.test_dir, "assets")
        self.output_base_dir = os.path.join(self.test_dir, "data")
        self.raw_clips_dir = os.path.join(
            self.assets_dir,
            self.project_name,
            "06_clips",
            "_raw",
        )
        os.makedirs(self.raw_clips_dir, exist_ok=True)
        with open(os.path.join(self.raw_clips_dir, "clip2.mp4"), "w", encoding="utf-8") as file:
            file.write("mock video")

        self.timeline_data = {
            "video_timeline": [
                {
                    "clip": "clip1.mp4",
                    "start_s": 0.0,
                    "end_s": 10.0,
                    "duration_s": 10.0,
                },
                {
                    "clip": "clip2.mp4",
                    "start_s": 10.0,
                    "end_s": 20.0,
                    "duration_s": 10.0,
                },
            ]
        }
        self.feedback_data = [
            {
                "clip_used": "clip1.mp4",
                "clip_start_s": 0.0,
                "clip_end_s": 10.0,
                "feedback_items": [
                    {
                        "timestamp": "00:02",
                        "category": "video",
                        "remark": "Use Vir's smile expression from 00:15 in this shot.",
                    }
                ],
            }
        ]
        with open(self.timeline_path, "w", encoding="utf-8") as file:
            json.dump(self.timeline_data, file)
        with open(self.feedback_path, "w", encoding="utf-8") as file:
            json.dump(self.feedback_data, file)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_segment_detection_requires_timestamp_and_reference_language(self):
        segment = {
            "feedback_items": [
                {"timestamp": "00:02", "remark": "Make Vir more playful."},
            ]
        }
        self.assertFalse(segment_needs_complex_reference_investigation(segment))

        segment["feedback_items"][0]["remark"] = "Use the same pose from 00:15."
        self.assertTrue(segment_needs_complex_reference_investigation(segment))

    def test_deterministic_plan_resolves_reference_timestamp_to_source_clip(self):
        result = build_deterministic_complex_feedback_plans(
            self.feedback_data,
            self.timeline_data,
            project_name=self.project_name,
            assets_dir=self.assets_dir,
            output_base_dir=self.output_base_dir,
            extract_frames=False,
        )

        self.assertEqual(1, len(result.plans))
        plan = result.plans[0]
        self.assertTrue(plan.requires_cross_reference)
        self.assertEqual("clip1.mp4", plan.target_clip)
        self.assertEqual("attach_reference_frame", plan.generation_strategy)
        self.assertEqual("clip2.mp4", plan.references[0].clip_used)
        self.assertEqual(5.0, plan.references[0].offset_s)
        self.assertEqual("expression_anchor", plan.references[0].usage)

    @mock.patch("src.workflows.complex_feedback_agent.extract_frame_at_offset")
    def test_investigation_attaches_resolved_reference_to_feedback_json(self, mock_extract):
        mock_extract.return_value = True

        result_path = investigate_complex_feedback_references(
            feedback_json_path=self.feedback_path,
            timeline_json_path=self.timeline_path,
            project_name=self.project_name,
            assets_dir=self.assets_dir,
            provider="openai",
            model="gpt-5.4-mini",
            output_base_dir=self.output_base_dir,
        )

        self.assertEqual(os.path.abspath(self.feedback_path), result_path)
        with open(self.feedback_path, "r", encoding="utf-8") as file:
            updated = json.load(file)

        refs = updated[0]["referenced_frames"]
        item_refs = updated[0]["feedback_items"][0]["referenced_frames"]
        self.assertEqual(1, len(refs))
        self.assertEqual(refs, item_refs)
        self.assertEqual("complex_feedback_agent", refs[0]["source"])
        self.assertEqual("clip2.mp4", refs[0]["clip_used"])
        self.assertTrue(updated[0]["complex_reference_plan"]["requires_cross_reference"])

    @mock.patch("src.workflows.complex_feedback_agent.extract_frame_at_offset")
    @mock.patch("src.workflows.complex_feedback_agent.run_complex_feedback_agent")
    def test_sdk_path_materializes_agent_plan_without_real_model_call(self, mock_agent, mock_extract):
        mock_extract.return_value = True
        mock_agent.return_value = ComplexFeedbackInvestigationResult(
            plans=[
                ComplexFeedbackPlan(
                    feedback_index=0,
                    target_clip="clip1.mp4",
                    target_start_s=0.0,
                    target_end_s=10.0,
                    remarks=["Use Vir's smile expression from 00:15 in this shot."],
                    requires_cross_reference=True,
                    references=[
                        ComplexReferencePlan(
                            timestamp="00:15",
                            reason="Use the earlier smile as an expression anchor.",
                            clip_used="clip2.mp4",
                            usage="expression_anchor",
                        )
                    ],
                    generation_strategy="attach_reference_frame",
                    confidence=0.8,
                )
            ]
        )

        investigate_complex_feedback_references(
            feedback_json_path=self.feedback_path,
            timeline_json_path=self.timeline_path,
            project_name=self.project_name,
            assets_dir=self.assets_dir,
            provider="openai",
            model="gpt-5.4-mini",
            output_base_dir=self.output_base_dir,
            use_agents_sdk=True,
        )

        mock_agent.assert_called_once()
        with open(self.feedback_path, "r", encoding="utf-8") as file:
            updated = json.load(file)
        self.assertEqual("expression_anchor", updated[0]["referenced_frames"][0]["usage"])
        self.assertTrue(updated[0]["referenced_frames"][0]["frame_path"].endswith("ref_00_15_clip2.jpg"))


if __name__ == "__main__":
    unittest.main()
