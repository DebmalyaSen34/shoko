import unittest

from src.clustering import (
    cluster_feedback_by_clip,
    find_matching_clip_occurrence,
    format_feedback_instruction,
    parse_timestamp_to_seconds,
)


class ClipMappingTests(unittest.TestCase):
    def setUp(self):
        self.timeline = [
            {"clip": "same.mp4", "start_s": 0.0, "end_s": 2.0},
            {"clip": "middle.mp4", "start_s": 2.0, "end_s": 4.0},
            {"clip": "same.mp4", "start_s": 4.25, "end_s": 6.0},
        ]

    def test_exact_cut_maps_to_later_occurrence(self):
        index, clip = find_matching_clip_occurrence(self.timeline, 2.0)
        self.assertEqual(1, index)
        self.assertEqual("middle.mp4", clip["clip"])

    def test_whole_second_boundary_maps_to_next_clip_starting_after_timestamp(self):
        timeline = [
            {"clip": "outgoing.mp4", "start_s": 40.0, "end_s": 44.01},
            {"clip": "incoming.mp4", "start_s": 44.02, "end_s": 48.0},
        ]

        index, clip = find_matching_clip_occurrence(timeline, 44.0)

        self.assertEqual(1, index)
        self.assertEqual("incoming.mp4", clip["clip"])

    def test_gap_maps_to_only_nearest_occurrence(self):
        index, _ = find_matching_clip_occurrence(self.timeline, 4.2)
        self.assertEqual(2, index)

    def test_final_endpoint_belongs_to_final_occurrence(self):
        index, _ = find_matching_clip_occurrence(self.timeline, 6.0)
        self.assertEqual(2, index)

    def test_timestamp_outside_tolerance_is_unmatched(self):
        self.assertEqual(
            (None, None),
            find_matching_clip_occurrence(self.timeline, 9.0),
        )

    def test_timestamp_parser_accepts_supported_formats(self):
        self.assertEqual(62.5, parse_timestamp_to_seconds("1:02.5"))
        self.assertEqual(3662.5, parse_timestamp_to_seconds("1:01:02.5"))
        self.assertIsNone(parse_timestamp_to_seconds("invalid"))


class FeedbackClusteringTests(unittest.TestCase):
    def setUp(self):
        self.timeline = [
            {
                "clip": "same.mp4",
                "start_s": 0.0,
                "end_s": 2.0,
                "start_tc": "00:00:00:00",
                "end_tc": "00:00:02:00",
                "duration_s": 2.0,
            },
            {
                "clip": "middle.mp4",
                "start_s": 2.0,
                "end_s": 4.0,
                "start_tc": "00:00:02:00",
                "end_tc": "00:00:04:00",
                "duration_s": 2.0,
            },
            {
                "clip": "same.mp4",
                "start_s": 4.0,
                "end_s": 6.0,
                "start_tc": "00:00:04:00",
                "end_tc": "00:00:06:00",
                "duration_s": 2.0,
            },
        ]

    def test_same_clip_clusters_by_lane_without_mixing(self):
        feedback = [
            {"timestamp": "00:01", "category": "video", "remark": "video one"},
            {"timestamp": "00:01.5", "category": "video", "remark": "video two"},
            {"timestamp": "00:01", "category": "audio", "remark": "audio one"},
            {"timestamp": "00:01.5", "category": "audio", "remark": "audio two"},
        ]

        clusters = cluster_feedback_by_clip(feedback, self.timeline)

        self.assertEqual(["video", "audio"], [c["category"] for c in clusters])
        self.assertEqual([2, 2], [len(c["feedback_items"]) for c in clusters])
        self.assertEqual(
            ["video one", "video two"],
            [item["remark"] for item in clusters[0]["feedback_items"]],
        )

    def test_both_feedback_enters_separate_video_and_audio_clusters(self):
        feedback = [
            {"timestamp": "00:03", "category": "both", "remark": "change both"},
        ]

        clusters = cluster_feedback_by_clip(feedback, self.timeline)

        self.assertEqual(["video", "audio"], [c["category"] for c in clusters])
        self.assertEqual("change both", clusters[0]["feedback_items"][0]["remark"])
        self.assertEqual("change both", clusters[1]["feedback_items"][0]["remark"])

    def test_repeated_filename_at_different_occurrences_stays_separate(self):
        feedback = [
            {"timestamp": "00:01", "category": "video", "remark": "first use"},
            {"timestamp": "00:05", "category": "video", "remark": "second use"},
        ]

        clusters = cluster_feedback_by_clip(feedback, self.timeline)

        self.assertEqual([0, 2], [c["clip_occurrence"] for c in clusters])

    def test_freeze_feedback_maps_to_final_occurrence(self):
        feedback = [
            {"timestamp": None, "category": "video", "remark": "Create freeze point"},
        ]

        clusters = cluster_feedback_by_clip(feedback, self.timeline)

        self.assertEqual(2, clusters[0]["clip_occurrence"])

    def test_invalid_timestamp_is_retained_as_unmatched(self):
        feedback = [
            {"timestamp": "bad", "category": "audio", "remark": "keep this"},
        ]

        clusters = cluster_feedback_by_clip(feedback, self.timeline)

        self.assertIsNone(clusters[0]["clip_occurrence"])
        self.assertIsNone(clusters[0]["matched_clip"])
        self.assertEqual("keep this", clusters[0]["feedback_items"][0]["remark"])

    def test_combined_instruction_keeps_timestamps_and_order(self):
        instruction = format_feedback_instruction(
            [
                {"timestamp": "00:01", "remark": "first"},
                {"timestamp": None, "remark": "second"},
            ]
        )

        self.assertEqual("- [00:01] first\n- [No Timestamp] second", instruction)


if __name__ == "__main__":
    unittest.main()
