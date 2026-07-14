import os
import json
import shutil
import tempfile
import unittest
from unittest import mock

real_exists = os.path.exists

from src.workflows.referenced_frames import analyze_and_extract_referenced_frames

class TestReferencedFramesWorkflow(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.feedback_path = os.path.join(self.test_dir, "feedback.json")
        self.timeline_path = os.path.join(self.test_dir, "timeline.json")
        self.assets_dir = os.path.join(self.test_dir, "assets")
        self.project_name = "test_project"
        self.output_base_dir = os.path.join(self.test_dir, "data")

        # Mock timeline
        self.timeline_data = {
            "video_timeline": [
                {
                    "clip": "clip1.mp4",
                    "start_s": 0.0,
                    "end_s": 10.0,
                    "start_tc": "00:00:00:00",
                    "end_tc": "00:00:10:00",
                    "duration_s": 10.0
                },
                {
                    "clip": "clip2.mp4",
                    "start_s": 10.0,
                    "end_s": 20.0,
                    "start_tc": "00:00:10:00",
                    "end_tc": "00:00:20:00",
                    "duration_s": 10.0
                }
            ]
        }

        with open(self.timeline_path, "w", encoding="utf-8") as f:
            json.dump(self.timeline_data, f)

        # Mock feedback
        self.feedback_data = [
            {
                "clip_used": "clip1.mp4",
                "clip_start_s": 0.0,
                "feedback_items": [
                    {
                        "timestamp": "00:02",
                        "category": "video",
                        "remark": "Vir is standing. Add Vir's smile used at 00:15."
                    }
                ]
            }
        ]

        with open(self.feedback_path, "w", encoding="utf-8") as f:
            json.dump(self.feedback_data, f)

        # Create mock raw clip files
        os.makedirs(os.path.join(self.assets_dir, self.project_name, "06_clips", "_raw"), exist_ok=True)
        self.clip2_path = os.path.join(self.assets_dir, self.project_name, "06_clips", "_raw", "clip2.mp4")
        with open(self.clip2_path, "w") as f:
            f.write("mock content")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @mock.patch("src.workflows.referenced_frames.generate_structured")
    @mock.patch("src.workflows.referenced_frames.subprocess.run")
    @mock.patch("src.workflows.referenced_frames.os.path.exists")
    def test_analyze_and_extract_referenced_frames_success(
        self, mock_exists, mock_run, mock_gen_structured
    ):
        # Configure mocks
        # Make os.path.exists return True for the clip path, and also true for the extracted frame
        def exists_side_effect(path):
            if "clip2.mp4" in path:
                return True
            if "referenced_frames" in path:
                # Mock that ffmpeg successfully created the file
                return True
            return real_exists(path)
        mock_exists.side_effect = exists_side_effect

        # Mock LLM structured response
        mock_gen_structured.return_value = {
            "referenced_timestamps": [
                {
                    "timestamp": "00:15",
                    "reason": "Vir's smile"
                }
            ]
        }

        # Mock subprocess.run returning success
        mock_completed = mock.Mock()
        mock_completed.returncode = 0
        mock_run.return_value = mock_completed

        # Run workflow
        analyze_and_extract_referenced_frames(
            feedback_json_path=self.feedback_path,
            timeline_json_path=self.timeline_path,
            project_name=self.project_name,
            assets_dir=self.assets_dir,
            client=mock.Mock(),
            provider="openai",
            model="gpt-4o",
            output_base_dir=self.output_base_dir
        )

        # Assertions
        with open(self.feedback_path, 'r', encoding='utf-8') as f:
            updated_feedback = json.load(f)

        segment = updated_feedback[0]
        self.assertIn("referenced_frames", segment)
        self.assertEqual(len(segment["referenced_frames"]), 1)
        ref_frame = segment["referenced_frames"][0]
        self.assertEqual(ref_frame["timestamp"], "00:15")
        self.assertEqual(ref_frame["reason"], "Vir's smile")
        self.assertEqual(ref_frame["clip_used"], "clip2.mp4")
        self.assertEqual(ref_frame["offset_s"], 5.0)  # 15.0s - 10.0s (clip2 start)

    @mock.patch("src.workflows.referenced_frames.generate_structured")
    def test_no_timestamp_regex_no_llm_call(self, mock_gen_structured):
        # Write feedback with no timestamps inside remark text
        no_ref_feedback = [
            {
                "clip_used": "clip1.mp4",
                "clip_start_s": 0.0,
                "feedback_items": [
                    {
                        "timestamp": "00:02",
                        "category": "video",
                        "remark": "Vir is standing. Make him look playful."
                    }
                ]
            }
        ]
        with open(self.feedback_path, "w", encoding="utf-8") as f:
            json.dump(no_ref_feedback, f)

        # Run workflow
        analyze_and_extract_referenced_frames(
            feedback_json_path=self.feedback_path,
            timeline_json_path=self.timeline_path,
            project_name=self.project_name,
            assets_dir=self.assets_dir,
            client=mock.Mock(),
            provider="openai",
            model="gpt-4o",
            output_base_dir=self.output_base_dir
        )

        # LLM should not be called because there are no matching timestamps in remark text
        mock_gen_structured.assert_not_called()

    @mock.patch("src.workflows.referenced_frames.generate_structured")
    @mock.patch("src.workflows.referenced_frames.subprocess.run")
    @mock.patch("src.workflows.referenced_frames.os.path.exists")
    def test_referenced_frames_are_deduped_when_workflow_reruns(
        self, mock_exists, mock_run, mock_gen_structured
    ):
        existing_ref = {
            "timestamp": "00:15",
            "reason": "Vir's smile",
            "frame_path": os.path.abspath(
                os.path.join(self.output_base_dir, self.project_name, "referenced_frames", "ref_00_15_clip2.jpg")
            ),
            "clip_used": "clip2.mp4",
            "offset_s": 5.0,
        }
        self.feedback_data[0]["referenced_frames"] = [existing_ref]
        self.feedback_data[0]["feedback_items"][0]["referenced_frames"] = [existing_ref]
        with open(self.feedback_path, "w", encoding="utf-8") as f:
            json.dump(self.feedback_data, f)

        def exists_side_effect(path):
            if "clip2.mp4" in path:
                return True
            if "referenced_frames" in path:
                return True
            return real_exists(path)
        mock_exists.side_effect = exists_side_effect

        mock_gen_structured.return_value = {
            "referenced_timestamps": [
                {"timestamp": "00:15", "reason": "Vir's smile"},
                {"timestamp": "00:15", "reason": "Vir's smile duplicate"},
            ]
        }
        mock_completed = mock.Mock()
        mock_completed.returncode = 0
        mock_run.return_value = mock_completed

        analyze_and_extract_referenced_frames(
            feedback_json_path=self.feedback_path,
            timeline_json_path=self.timeline_path,
            project_name=self.project_name,
            assets_dir=self.assets_dir,
            client=mock.Mock(),
            provider="openai",
            model="gpt-4o",
            output_base_dir=self.output_base_dir,
        )

        with open(self.feedback_path, "r", encoding="utf-8") as f:
            updated_feedback = json.load(f)

        segment_refs = updated_feedback[0]["referenced_frames"]
        item_refs = updated_feedback[0]["feedback_items"][0]["referenced_frames"]
        self.assertEqual(1, len(segment_refs))
        self.assertEqual(1, len(item_refs))
        self.assertEqual(existing_ref["frame_path"], segment_refs[0]["frame_path"])

if __name__ == "__main__":
    unittest.main()
