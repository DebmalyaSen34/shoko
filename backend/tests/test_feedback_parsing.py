import os
import json
import shutil
import tempfile
import unittest
import zipfile
from unittest import mock

from src.workflows.feedback_parsing import (
    parse_and_align_feedback,
    read_xlsx_file,
    extract_file_content
)

class TestFeedbackParsingWorkflow(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.timeline_path = os.path.join(self.test_dir, "timeline.json")
        self.feedback_txt_path = os.path.join(self.test_dir, "feedback.txt")
        self.feedback_csv_path = os.path.join(self.test_dir, "feedback.csv")
        self.feedback_xlsx_path = os.path.join(self.test_dir, "feedback.xlsx")
        self.output_base_dir = os.path.join(self.test_dir, "data")

        # Create a mock timeline.json content
        self.mock_timeline = {
            "video_timeline": [
                {
                    "clip": "clip1.mp4",
                    "start_tc": "00:00:00:00",
                    "end_tc": "00:00:10:00",
                    "start_s": 0.0,
                    "end_s": 10.0,
                    "duration_s": 10.0
                },
                {
                    "clip": "clip2.mp4",
                    "start_tc": "00:00:10:00",
                    "end_tc": "00:00:20:00",
                    "start_s": 10.0,
                    "end_s": 20.0,
                    "duration_s": 10.0
                }
            ],
            "audio_timeline": {
                "dedicated_audio_tracks": [
                    {
                        "clip": "audio1.mp3",
                        "start_tc": "00:00:00:00",
                        "end_tc": "00:00:10:00",
                        "start_s": 0.0,
                        "end_s": 10.0,
                        "duration_s": 10.0
                    }
                ]
            }
        }

        with open(self.timeline_path, "w", encoding="utf-8") as f:
            json.dump(self.mock_timeline, f)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_read_txt_file(self):
        with open(self.feedback_txt_path, "w", encoding="utf-8") as f:
            f.write("00:12\tSome feedback remark")
        
        content = extract_file_content(self.feedback_txt_path)
        self.assertEqual(content, "00:12\tSome feedback remark")

    def test_read_csv_file(self):
        with open(self.feedback_csv_path, "w", encoding="utf-8") as f:
            f.write("00:12,Some remark\n00:15,Other remark\n")
        
        content = extract_file_content(self.feedback_csv_path)
        self.assertEqual(content, "00:12\tSome remark\n00:15\tOther remark")

    def test_read_xlsx_file_custom_parser(self):
        # Create a mock zip-based Excel .xlsx file structure
        shared_strings_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
        <sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" count="2" uniqueCount="2">
            <si><t>Shared String A</t></si>
            <si><t>Shared String B</t></si>
        </sst>
        """
        
        sheet1_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
        <worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
            <sheetData>
                <row r="1">
                    <c r="A1" t="s"><v>0</v></c>
                    <c r="B1"><v>123.45</v></c>
                </row>
                <row r="2">
                    <c r="A2" t="s"><v>1</v></c>
                    <c r="B2" t="s"><v>0</v></c>
                </row>
            </sheetData>
        </worksheet>
        """

        with zipfile.ZipFile(self.feedback_xlsx_path, "w") as zf:
            zf.writestr("xl/sharedStrings.xml", shared_strings_xml)
            zf.writestr("xl/worksheets/sheet1.xml", sheet1_xml)

        content = read_xlsx_file(self.feedback_xlsx_path)
        expected_rows = [
            "Shared String A\t123.45",
            "Shared String B\tShared String A"
        ]
        self.assertEqual(content, "\n".join(expected_rows))

    @mock.patch("src.workflows.feedback_parsing.generate_structured")
    def test_parse_and_align_feedback_option_a_grouped(self, mock_gen_structured):
        # Arrange
        # Mock LLM output
        mock_gen_structured.return_value = {
            "remarks": [
                {
                    "timestamp": "00:02",
                    "category": "audio",
                    "remark": "Make dialogue clearer"
                },
                {
                    "timestamp": "00:12",
                    "category": "video",
                    "remark": "Vir should look playful"
                },
                {
                    "timestamp": "00:14",
                    "category": "video",
                    "remark": "Zoom in reaction"
                }
            ]
        }

        # Write dummy feedback text file
        with open(self.feedback_txt_path, "w", encoding="utf-8") as f:
            f.write("00:02\tMake dialogue clearer\n00:12\tVir should look playful\n00:14\tZoom in reaction")

        project_name = "test_feedback_project"
        mock_client = mock.MagicMock()

        # Act
        result_path = parse_and_align_feedback(
            feedback_file_path=self.feedback_txt_path,
            timeline_json_path=self.timeline_path,
            project_name=project_name,
            openai_client=mock_client,
            output_base_dir=self.output_base_dir
        )

        # Assert
        expected_output_path = os.path.abspath(
            os.path.join(self.output_base_dir, project_name, "feedback.json")
        )
        self.assertEqual(result_path, expected_output_path)
        self.assertTrue(os.path.exists(result_path))

        with open(result_path, "r", encoding="utf-8") as f:
            output_data = json.load(f)

        # We expect two segment blocks:
        # - Segment 1: clip1.mp4 (has audio_used="audio1.mp3", contains feedback_items=[Make dialogue clearer])
        # - Segment 2: clip2.mp4 (has audio_used=None, contains feedback_items=[Vir should look playful, Zoom in reaction])
        self.assertEqual(len(output_data), 2)

        # Segment for clip1.mp4 (timestamp 00:02)
        seg1 = next(s for s in output_data if s.get("clip_used") == "clip1.mp4")
        self.assertEqual(seg1["audio_used"], "audio1.mp3")
        self.assertEqual(seg1["previous_clip"], None)
        self.assertEqual(len(seg1["feedback_items"]), 1)
        self.assertEqual(seg1["feedback_items"][0]["timestamp"], "00:02")
        self.assertEqual(seg1["feedback_items"][0]["category"], "audio")
        self.assertEqual(seg1["feedback_items"][0]["remark"], "Make dialogue clearer")

        # Segment for clip2.mp4 (timestamps 00:12 and 00:14)
        seg2 = next(s for s in output_data if s.get("clip_used") == "clip2.mp4")
        self.assertEqual(seg2["audio_used"], None)
        self.assertEqual(seg2["previous_clip"], "clip1.mp4")
        self.assertEqual(len(seg2["feedback_items"]), 2)
        self.assertEqual(seg2["feedback_items"][0]["timestamp"], "00:12")
        self.assertEqual(seg2["feedback_items"][0]["category"], "video")
        self.assertEqual(seg2["feedback_items"][1]["timestamp"], "00:14")
        self.assertEqual(seg2["feedback_items"][1]["category"], "video")

if __name__ == "__main__":
    unittest.main()
