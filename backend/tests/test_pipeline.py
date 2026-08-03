import json
import os
import tempfile
import unittest
from unittest import mock

import main


class ClusteredPipelineTests(unittest.TestCase):
    def test_batch_size_must_be_positive_before_api_initialization(self):
        with self.assertRaisesRegex(ValueError, "batch_size must be positive"):
            main.run_pipeline(
                feedback_path="unused-feedback.json",
                timeline_path="unused-timeline.json",
                assets_dir="unused-assets",
                output_json="unused-output.json",
                output_report="unused-report.md",
                batch_size=0,
            )

    def test_pipeline_batches_video_and_both_feedback_but_ignores_audio(self):
        feedback = [
            {"timestamp": "00:01", "category": "video", "remark": "video one"},
            {"timestamp": "00:01.2", "category": "audio", "remark": "audio only"},
            {"timestamp": "00:01.5", "category": "both", "remark": "visual and sound"},
        ]
        timeline = {
            "frame_size": "1080x1920",
            "video_timeline": [
                {
                    "clip": "clip.mp4",
                    "start_s": 0.0,
                    "end_s": 2.0,
                    "start_tc": "00:00:00:00",
                    "end_tc": "00:00:02:00",
                    "duration_s": 2.0,
                }
            ]
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            feedback_path = os.path.join(temp_dir, "feedback.json")
            timeline_path = os.path.join(temp_dir, "timeline.json")
            output_json = os.path.join(temp_dir, "output", "result.json")
            output_report = os.path.join(temp_dir, "output", "report.md")
            os.makedirs(os.path.dirname(output_json), exist_ok=True)
            with open(feedback_path, "w", encoding="utf-8") as file:
                json.dump(feedback, file)
            with open(timeline_path, "w", encoding="utf-8") as file:
                json.dump(timeline, file)
            with open(os.path.join(os.path.dirname(output_json), "prompt_lessons.json"), "w", encoding="utf-8") as file:
                json.dump(
                    {
                        "schema_version": 1,
                        "lessons": [
                            {
                                "id": "lesson-continuity",
                                "scope": "project",
                                "clip_key": None,
                                "category": "continuity_error",
                                "lesson": "Preserve wardrobe when feedback mentions continuity.",
                                "source_feedback_ids": [],
                                "confidence": 0.9,
                                "positive_examples": [],
                                "negative_examples": [],
                                "created_at": "2026-01-01T00:00:00Z",
                                "updated_at": "2026-01-01T00:00:00Z",
                                "archived": False,
                            }
                        ],
                    },
                    file,
                )
            with open(os.path.join(os.path.dirname(output_json), "prompt_eval_cases.json"), "w", encoding="utf-8") as file:
                json.dump(
                    {
                        "schema_version": 1,
                        "cases": [
                            {
                                "id": "case-continuity",
                                "name": "Continuity wardrobe preservation",
                                "source_feedback_id": "feedback-1",
                                "clip_index": 0,
                                "clip_key": "clip.mp4::0",
                                "input": {
                                    "feedback_items": [{"remark": "video one"}],
                                    "clip_summary": "",
                                    "selected_assets": ["character.png"],
                                },
                                "expected_behavior": ["Must preserve wardrobe continuity."],
                                "failure_categories": ["continuity_error"],
                                "enabled": True,
                                "created_at": "2026-01-01T00:00:00Z",
                                "updated_at": "2026-01-01T00:00:00Z",
                            }
                        ],
                    },
                    file,
                )

            with (
                mock.patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}),
                mock.patch("main.genai.Client", return_value=object()),
                mock.patch(
                    "main.scan_visual_reference_assets",
                    return_value=["character.png"],
                ),
                mock.patch(
                    "main.generate_video_prompts_batch",
                    return_value=[
                        {
                            "cluster_id": 0,
                            "initial_frame_prompt": "initial frame result",
                            "initial_frame_image_path": os.path.join(
                                temp_dir,
                                "output",
                                "initial_frames",
                                "cluster_0_initial_frame.png",
                            ),
                            "selected_assets": ["character.png"],
                            "prompt_format": "plain_text",
                            "reference_legend": "image 1 — character.png\noriginal clip reference images",
                            "video_model_prompt": "video result",
                            "explanation": "video",
                            "status": "success",
                            "applied_prompt_lessons": [
                                {
                                    "id": "lesson-continuity",
                                    "scope": "project",
                                    "category": "continuity_error",
                                    "lesson": "Preserve wardrobe when feedback mentions continuity.",
                                }
                            ],
                        }
                    ],
                ) as batch_generator,
                mock.patch("main.write_markdown_report"),
            ):
                main.run_pipeline(
                    feedback_path=feedback_path,
                    timeline_path=timeline_path,
                    assets_dir=temp_dir,
                    output_json=output_json,
                    output_report=output_report,
                )

            self.assertEqual(1, batch_generator.call_count)
            passed_clusters = batch_generator.call_args.kwargs["clusters"]
            self.assertTrue(all(c["category"] == "video" for c in passed_clusters))
            self.assertEqual(5, batch_generator.call_args.kwargs["batch_size"])
            self.assertEqual(
                "lesson-continuity",
                batch_generator.call_args.kwargs["prompt_lessons_by_cluster"][0][0]["id"],
            )
            self.assertEqual(
                "case-continuity",
                batch_generator.call_args.kwargs["prompt_eval_cases_by_cluster"][0][0]["id"],
            )
            self.assertEqual("1080x1920", passed_clusters[0]["frame_size"])
            self.assertEqual(
                ["video one", "visual and sound"],
                [item["remark"] for item in passed_clusters[0]["feedback_items"]],
            )

            with open(output_json, "r", encoding="utf-8") as file:
                results = json.load(file)
            self.assertEqual(1, len(results))
            self.assertEqual("video", results[0]["category"])
            self.assertEqual(2, len(results[0]["feedback_items"]))
            self.assertEqual("success", results[0]["status"])
            self.assertEqual(
                "lesson-continuity",
                results[0]["applied_prompt_lessons"][0]["id"],
            )
            self.assertEqual("plain_text", results[0]["prompt_format"])
            self.assertEqual("initial frame result", results[0]["initial_frame_prompt"])
            self.assertTrue(results[0]["initial_frame_image_path"].endswith(
                "cluster_0_initial_frame.png"
            ))
            self.assertIn("original clip reference images", results[0]["reference_legend"])
            self.assertTrue(
                batch_generator.call_args.kwargs["initial_frames_dir"].endswith(
                    os.path.join("output", "initial_frames")
                )
            )

    def test_legacy_pipeline_invokes_complex_feedback_enrichment_when_enabled(self):
        feedback = [
            {
                "timestamp": "00:01",
                "category": "video",
                "remark": "Use Vir's smile expression from 00:15 in this shot.",
            }
        ]
        timeline = {
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

        with tempfile.TemporaryDirectory() as temp_dir:
            feedback_path = os.path.join(temp_dir, "feedback.json")
            timeline_path = os.path.join(temp_dir, "timeline.json")
            output_json = os.path.join(temp_dir, "project-a", "output.json")
            output_report = os.path.join(temp_dir, "project-a", "report.md")
            with open(feedback_path, "w", encoding="utf-8") as file:
                json.dump(feedback, file)
            with open(timeline_path, "w", encoding="utf-8") as file:
                json.dump(timeline, file)

            enriched_feedback = [
                {
                    **feedback[0],
                    "referenced_frames": [
                        {
                            "timestamp": "00:15",
                            "frame_path": os.path.join(temp_dir, "ref.jpg"),
                            "clip_used": "clip2.mp4",
                            "usage": "expression_anchor",
                        }
                    ],
                }
            ]

            with (
                mock.patch.dict(
                    os.environ,
                    {
                        "OPENAI_API_KEY": "test-key",
                        "LOKA_USE_COMPLEX_FEEDBACK_AGENT": "1",
                    },
                ),
                mock.patch("main.OpenAI", return_value=object()),
                mock.patch("main.scan_visual_reference_assets", return_value=[]),
                mock.patch(
                    "main.enrich_feedback_items_with_complex_references",
                    return_value=(enriched_feedback, True, "agent"),
                ) as enrich_refs,
                mock.patch(
                    "main.generate_video_prompts_batch",
                    return_value=[
                        {
                            "cluster_id": 0,
                            "selected_assets": [],
                            "prompt_format": "plain_text",
                            "reference_legend": "",
                            "video_model_prompt": "video result",
                            "explanation": "video",
                            "status": "success",
                        }
                    ],
                ) as batch_generator,
                mock.patch("main.write_markdown_report"),
            ):
                main.run_pipeline(
                    feedback_path=feedback_path,
                    timeline_path=timeline_path,
                    assets_dir=os.path.join(temp_dir, "assets", "project-a"),
                    output_json=output_json,
                    output_report=output_report,
                    provider="openai",
                )

            enrich_refs.assert_called_once()
            self.assertEqual("openai", enrich_refs.call_args.kwargs["provider"])
            passed_cluster = batch_generator.call_args.kwargs["clusters"][0]
            self.assertEqual("expression_anchor", passed_cluster["feedback_items"][0]["referenced_frames"][0]["usage"])

    def test_audio_only_feedback_makes_no_batch_generation_call(self):
        feedback = [
            {"timestamp": "00:01", "category": "audio", "remark": "audio only"},
        ]
        timeline = {
            "video_timeline": [
                {
                    "clip": "clip.mp4",
                    "start_s": 0.0,
                    "end_s": 2.0,
                    "start_tc": "00:00:00:00",
                    "end_tc": "00:00:02:00",
                    "duration_s": 2.0,
                }
            ]
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            feedback_path = os.path.join(temp_dir, "feedback.json")
            timeline_path = os.path.join(temp_dir, "timeline.json")
            output_json = os.path.join(temp_dir, "output", "result.json")
            output_report = os.path.join(temp_dir, "output", "report.md")
            with open(feedback_path, "w", encoding="utf-8") as file:
                json.dump(feedback, file)
            with open(timeline_path, "w", encoding="utf-8") as file:
                json.dump(timeline, file)

            with (
                mock.patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}),
                mock.patch("main.genai.Client", return_value=object()),
                mock.patch("main.scan_visual_reference_assets", return_value=[]),
                mock.patch("main.generate_video_prompts_batch", return_value=[]) as batch_generator,
                mock.patch("main.write_markdown_report"),
            ):
                main.run_pipeline(
                    feedback_path=feedback_path,
                    timeline_path=timeline_path,
                    assets_dir=temp_dir,
                    output_json=output_json,
                    output_report=output_report,
                )

            batch_generator.assert_not_called()
            with open(output_json, "r", encoding="utf-8") as file:
                self.assertEqual([], json.load(file))

    def test_index_run_expands_to_same_group_video_siblings(self):
        feedback = [
            {
                "timestamp": "00:45",
                "category": "video",
                "remark": "Show Vir grasping his mother's hand.",
                "group_id": "output23::7",
                "clip_used": "output (23).mp4",
                "clip_occurrence": 7,
                "sibling_raw_indexes": [1],
            },
            {
                "timestamp": "00:46",
                "category": "video",
                "remark": "Have mother begin to get angry and look down.",
                "group_id": "output23::7",
                "clip_used": "output (23).mp4",
                "clip_occurrence": 7,
                "sibling_raw_indexes": [0],
            },
        ]
        timeline = {
            "video_timeline": [
                {
                    "clip": "output (23).mp4",
                    "start_s": 44.0,
                    "end_s": 48.0,
                    "start_tc": "00:00:44:00",
                    "end_tc": "00:00:48:00",
                    "duration_s": 4.0,
                }
            ]
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            feedback_path = os.path.join(temp_dir, "feedback.json")
            timeline_path = os.path.join(temp_dir, "timeline.json")
            output_json = os.path.join(temp_dir, "output", "result.json")
            output_report = os.path.join(temp_dir, "output", "report.md")
            with open(feedback_path, "w", encoding="utf-8") as file:
                json.dump(feedback, file)
            with open(timeline_path, "w", encoding="utf-8") as file:
                json.dump(timeline, file)

            with (
                mock.patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}),
                mock.patch("main.genai.Client", return_value=object()),
                mock.patch("main.scan_visual_reference_assets", return_value=[]),
                mock.patch(
                    "main.generate_video_prompts_batch",
                    return_value=[
                        {
                            "cluster_id": 0,
                            "selected_assets": [],
                            "prompt_format": "plain_text",
                            "reference_legend": "",
                            "video_model_prompt": "combined prompt",
                            "explanation": "combined",
                            "status": "success",
                        }
                    ],
                ) as batch_generator,
                mock.patch("main.write_markdown_report"),
            ):
                main.run_pipeline(
                    feedback_path=feedback_path,
                    timeline_path=timeline_path,
                    assets_dir=temp_dir,
                    output_json=output_json,
                    output_report=output_report,
                    index=1,
                )

            passed_clusters = batch_generator.call_args.kwargs["clusters"]
            self.assertEqual(1, len(passed_clusters))
            self.assertEqual(
                [
                    "Show Vir grasping his mother's hand.",
                    "Have mother begin to get angry and look down.",
                ],
                [item["remark"] for item in passed_clusters[0]["feedback_items"]],
            )

    def test_index_run_does_not_pull_audio_only_sibling_into_video_cluster(self):
        feedback = [
            {
                "timestamp": "00:45",
                "category": "video",
                "remark": "Show Vir grasping his mother's hand.",
                "group_id": "output23::7",
                "clip_used": "output (23).mp4",
                "clip_occurrence": 7,
                "sibling_raw_indexes": [1],
            },
            {
                "timestamp": "00:46",
                "category": "audio",
                "remark": "Make the line louder.",
                "group_id": "output23::7",
                "clip_used": "output (23).mp4",
                "clip_occurrence": 7,
                "sibling_raw_indexes": [0],
            },
        ]
        timeline = {
            "video_timeline": [
                {
                    "clip": "output (23).mp4",
                    "start_s": 44.0,
                    "end_s": 48.0,
                    "start_tc": "00:00:44:00",
                    "end_tc": "00:00:48:00",
                    "duration_s": 4.0,
                }
            ]
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            feedback_path = os.path.join(temp_dir, "feedback.json")
            timeline_path = os.path.join(temp_dir, "timeline.json")
            output_json = os.path.join(temp_dir, "output", "result.json")
            output_report = os.path.join(temp_dir, "output", "report.md")
            with open(feedback_path, "w", encoding="utf-8") as file:
                json.dump(feedback, file)
            with open(timeline_path, "w", encoding="utf-8") as file:
                json.dump(timeline, file)

            with (
                mock.patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}),
                mock.patch("main.genai.Client", return_value=object()),
                mock.patch("main.scan_visual_reference_assets", return_value=[]),
                mock.patch(
                    "main.generate_video_prompts_batch",
                    return_value=[
                        {
                            "cluster_id": 0,
                            "selected_assets": [],
                            "prompt_format": "plain_text",
                            "reference_legend": "",
                            "video_model_prompt": "video only prompt",
                            "explanation": "video only",
                            "status": "success",
                        }
                    ],
                ) as batch_generator,
                mock.patch("main.write_markdown_report"),
            ):
                main.run_pipeline(
                    feedback_path=feedback_path,
                    timeline_path=timeline_path,
                    assets_dir=temp_dir,
                    output_json=output_json,
                    output_report=output_report,
                    index=0,
                )

            passed_clusters = batch_generator.call_args.kwargs["clusters"]
            self.assertEqual(
                ["Show Vir grasping his mother's hand."],
                [item["remark"] for item in passed_clusters[0]["feedback_items"]],
            )

    def test_pipeline_can_initialize_openai_provider(self):
        feedback = [
            {"timestamp": "00:01", "category": "video", "remark": "video one"},
        ]
        timeline = {
            "frame_size": "1080x1920",
            "video_timeline": [
                {
                    "clip": "clip.mp4",
                    "start_s": 0.0,
                    "end_s": 2.0,
                    "start_tc": "00:00:00:00",
                    "end_tc": "00:00:02:00",
                    "duration_s": 2.0,
                }
            ]
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            feedback_path = os.path.join(temp_dir, "feedback.json")
            timeline_path = os.path.join(temp_dir, "timeline.json")
            output_json = os.path.join(temp_dir, "output", "result.json")
            output_report = os.path.join(temp_dir, "output", "report.md")
            with open(feedback_path, "w", encoding="utf-8") as file:
                json.dump(feedback, file)
            with open(timeline_path, "w", encoding="utf-8") as file:
                json.dump(timeline, file)

            openai_client = object()
            with (
                mock.patch.dict(
                    os.environ,
                    {"OPENAI_API_KEY": "openai-key"},
                    clear=True,
                ),
                mock.patch("main.OpenAI", return_value=openai_client) as openai_cls,
                mock.patch("main.genai.Client") as gemini_cls,
                mock.patch("main.scan_visual_reference_assets", return_value=[]),
                mock.patch(
                    "main.generate_video_prompts_batch",
                    return_value=[
                        {
                            "cluster_id": 0,
                            "initial_frame_prompt": "initial frame result",
                            "selected_assets": [],
                            "prompt_format": "plain_text",
                            "reference_legend": "@video1 — original clip",
                            "video_model_prompt": "video result",
                            "explanation": "video",
                            "status": "success",
                        }
                    ],
                ) as batch_generator,
                mock.patch("main.write_markdown_report"),
            ):
                main.run_pipeline(
                    feedback_path=feedback_path,
                    timeline_path=timeline_path,
                    assets_dir=temp_dir,
                    output_json=output_json,
                    output_report=output_report,
                    provider="openai",
                )

            openai_cls.assert_called_once_with(api_key="openai-key")
            gemini_cls.assert_not_called()
            self.assertIs(openai_client, batch_generator.call_args.kwargs["client"])
            self.assertEqual("openai", batch_generator.call_args.kwargs["provider"])

    def test_pipeline_adds_continuity_frame_to_target_clusters(self):
        feedback = [
            {"timestamp": "00:01", "category": "video", "remark": "video one"},
        ]
        timeline = {
            "video_timeline": [
                {
                    "clip": "clip.mp4",
                    "start_s": 0.0,
                    "end_s": 2.0,
                    "start_tc": "00:00:00:00",
                    "end_tc": "00:00:02:00",
                    "duration_s": 2.0,
                }
            ]
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            feedback_path = os.path.join(temp_dir, "feedback.json")
            timeline_path = os.path.join(temp_dir, "timeline.json")
            output_json = os.path.join(temp_dir, "output", "result.json")
            output_report = os.path.join(temp_dir, "output", "report.md")
            continuity_frame = os.path.join(temp_dir, "last_frame.jpg")
            with open(feedback_path, "w", encoding="utf-8") as file:
                json.dump(feedback, file)
            with open(timeline_path, "w", encoding="utf-8") as file:
                json.dump(timeline, file)
            with open(continuity_frame, "wb") as file:
                file.write(b"frame")

            with (
                mock.patch.dict(os.environ, {"OPENAI_API_KEY": "openai-key"}, clear=True),
                mock.patch("main.OpenAI", return_value=object()),
                mock.patch("main.scan_visual_reference_assets", return_value=[]),
                mock.patch(
                    "main.generate_video_prompts_batch",
                    return_value=[
                        {
                            "cluster_id": 0,
                            "initial_frame_prompt": "continuity prompt",
                            "initial_frame_image_path": continuity_frame,
                            "selected_assets": [],
                            "prompt_format": "plain_text",
                            "reference_legend": "",
                            "video_model_prompt": "video result",
                            "explanation": "video",
                            "status": "success",
                        }
                    ],
                ) as batch_generator,
                mock.patch("main.write_markdown_report"),
            ):
                main.run_pipeline(
                    feedback_path=feedback_path,
                    timeline_path=timeline_path,
                    assets_dir=temp_dir,
                    output_json=output_json,
                    output_report=output_report,
                    provider="openai",
                    continuity_frame_path=continuity_frame,
                    continuity_note="Use previous last frame.",
                )

            passed_cluster = batch_generator.call_args.kwargs["clusters"][0]
            self.assertEqual(os.path.abspath(continuity_frame), passed_cluster["continuity_reference_frame_path"])
            self.assertEqual("Use previous last frame.", passed_cluster["continuity_reference_note"])

    def test_report_lists_every_feedback_item_in_cluster(self):
        results = [
            {
                "category": "video",
                "feedback_items": [
                    {"timestamp": "00:01", "remark": "first change"},
                    {"timestamp": "00:01.5", "remark": "second change"},
                ],
                "matched_clip": "clip.mp4",
                "clip_start_tc": "00:00:00:00",
                "clip_end_tc": "00:00:02:00",
                "clip_duration_s": 2.0,
                "selected_assets": [],
                "prompt_format": "plain_text",
                "reference_legend": "original clip reference images",
                "initial_frame_prompt": "initial still frame",
                "initial_frame_image_path": "/tmp/initial-frame.png",
                "video_model_prompt": "combined prompt",
                "explanation": "combined explanation",
            }
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = os.path.join(temp_dir, "report.md")
            main.write_markdown_report(results, report_path)
            with open(report_path, "r", encoding="utf-8") as file:
                report = file.read()

        self.assertIn("[00:01] first change", report)
        self.assertIn("[00:01.5] second change", report)
        self.assertEqual(1, report.count("combined prompt"))
        self.assertIn("initial still frame", report)
        self.assertIn("/tmp/initial-frame.png", report)
        self.assertIn("Seedance 2.0", report)
        self.assertIn("original clip reference images", report)

    def test_default_feedback_and_timeline_paths_belong_to_episode_one(self):
        self.assertEqual("data/feedback/feedback.json", main.DEFAULT_FEEDBACK_JSON_PATH)
        self.assertEqual("data/timeline/output.json", main.DEFAULT_TIMELINE_PATH)
        self.assertNotIn("output_1.json", main.DEFAULT_TIMELINE_PATH)

    def test_cli_args_support_single_index_test_runs(self):
        args = main.parse_args([
            "--feedback-path", "feedback.json",
            "--timeline-path", "timeline.json",
            "--assets-dir", "assets",
            "--output-json", "one.json",
            "--output-report", "one.md",
            "--provider", "openai",
            "--index", "2",
            "--batch-size", "1",
        ])

        self.assertEqual("feedback.json", args.feedback_path)
        self.assertEqual("timeline.json", args.timeline_path)
        self.assertEqual("assets", args.assets_dir)
        self.assertEqual("one.json", args.output_json)
        self.assertEqual("one.md", args.output_report)
        self.assertEqual("openai", args.provider)
        self.assertEqual(2, args.index)
        self.assertEqual(1, args.batch_size)

    def test_cli_args_support_agentic_resume_controls(self):
        args = main.parse_args([
            "--agentic",
            "--project-package-path", "handoff.zip",
            "--feedback-path", "feedback.xlsx",
            "--project-name", "project-red-and-green",
            "--assets-dir", "assets",
            "--provider", "openai",
            "--agentic-stage", "plan",
            "--force-stage", "references",
            "--force-stage", "plan",
        ])

        self.assertTrue(args.agentic)
        self.assertEqual("handoff.zip", args.project_package_path)
        self.assertEqual("plan", args.agentic_stage)
        self.assertIsNone(args.agentic_from)
        self.assertEqual(["references", "plan"], args.force_stage)

        resume_args = main.parse_args([
            "--agentic",
            "--project-name", "project-red-and-green",
            "--assets-dir", "assets",
            "--agentic-from", "plan",
        ])

        self.assertEqual("plan", resume_args.agentic_from)
        self.assertIsNone(resume_args.agentic_stage)

    @mock.patch("main.extract_timeline_from_project")
    @mock.patch("main.parse_and_align_feedback")
    @mock.patch("main.analyze_and_extract_referenced_frames")
    @mock.patch("main.plan_generation_workflow")
    @mock.patch("main.generate_video_prompts_from_plan")
    def test_run_agentic_loop_orchestrates_correctly(
        self, mock_gen_prompts, mock_plan, mock_ref_frames, mock_align, mock_timeline
    ):
        mock_timeline.return_value = "timeline.json"
        mock_align.return_value = "feedback.json"
        mock_gen_prompts.return_value = "video_prompts.json"

        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                mock.patch.dict(os.environ, {"OPENAI_API_KEY": "openai-key"}),
                mock.patch("main.genai.Client") as mock_gemini_cls,
                mock.patch("main.OpenAI") as mock_openai_cls,
            ):
                main.run_agentic_loop(
                    prproj_path="project.prproj",
                    raw_feedback_path="feedback.txt",
                    project_name="test_project",
                    assets_dir="assets",
                    output_base_dir=temp_dir,
                    provider="openai"
                )

        mock_timeline.assert_called_once_with(
            prproj_path="project.prproj",
            project_name="test_project",
            output_base_dir=temp_dir
        )
        mock_align.assert_called_once()
        mock_ref_frames.assert_called_once()
        mock_plan.assert_called_once()
        mock_gen_prompts.assert_called_once()

    @mock.patch("main.extract_timeline_from_project")
    @mock.patch("main.parse_and_align_feedback")
    @mock.patch("main.analyze_and_extract_referenced_frames")
    @mock.patch("main.plan_generation_workflow")
    @mock.patch("main.generate_video_prompts_from_plan")
    def test_run_agentic_loop_can_run_only_plan_stage(
        self, mock_gen_prompts, mock_plan, mock_ref_frames, mock_align, mock_timeline
    ):
        with tempfile.TemporaryDirectory() as temp_dir:
            project_name = "test_project"
            project_dir = os.path.join(temp_dir, project_name)
            os.makedirs(project_dir)
            feedback_path = os.path.join(project_dir, "feedback.json")
            with open(feedback_path, "w", encoding="utf-8") as file:
                json.dump([{"clip_used": "clip.mp4", "feedback_items": []}], file)

            plan_path = os.path.join(project_dir, "generation_plan.json")
            mock_plan.return_value = plan_path

            with (
                mock.patch.dict(os.environ, {"OPENAI_API_KEY": "openai-key"}, clear=True),
                mock.patch("main.OpenAI", return_value=object()),
            ):
                result = main.run_agentic_loop(
                    prproj_path=None,
                    raw_feedback_path=None,
                    project_name=project_name,
                    assets_dir="assets",
                    output_base_dir=temp_dir,
                    provider="openai",
                    stage="plan",
                )

            self.assertEqual(plan_path, result)
            mock_timeline.assert_not_called()
            mock_align.assert_not_called()
            mock_ref_frames.assert_not_called()
            mock_plan.assert_called_once_with(
                feedback_json_path=feedback_path,
                project_name=project_name,
                openai_client=mock.ANY,
                openai_model="gpt-5.4-mini",
                output_base_dir=temp_dir,
            )
            mock_gen_prompts.assert_not_called()

            manifest_path = os.path.join(project_dir, "agent_run_manifest.json")
            with open(manifest_path, "r", encoding="utf-8") as file:
                manifest = json.load(file)
            self.assertEqual("plan", manifest["final_output"])
            self.assertEqual("ran", manifest["stages"]["plan"]["status"])
            self.assertEqual("skipped", manifest["stages"]["timeline"]["status"])

    @mock.patch("main.extract_timeline_from_project")
    @mock.patch("main.parse_and_align_feedback")
    @mock.patch("main.analyze_and_extract_referenced_frames")
    @mock.patch("main.plan_generation_workflow")
    @mock.patch("main.generate_video_prompts_from_plan")
    def test_run_agentic_loop_can_resume_from_plan_stage(
        self, mock_gen_prompts, mock_plan, mock_ref_frames, mock_align, mock_timeline
    ):
        with tempfile.TemporaryDirectory() as temp_dir:
            project_name = "test_project"
            project_dir = os.path.join(temp_dir, project_name)
            os.makedirs(project_dir)
            feedback_path = os.path.join(project_dir, "feedback.json")
            plan_path = os.path.join(project_dir, "generation_plan.json")
            prompts_path = os.path.join(project_dir, "video_prompts.json")
            with open(feedback_path, "w", encoding="utf-8") as file:
                json.dump([], file)
            with open(plan_path, "w", encoding="utf-8") as file:
                json.dump([], file)

            mock_plan.return_value = plan_path
            mock_gen_prompts.return_value = prompts_path

            with (
                mock.patch.dict(os.environ, {"OPENAI_API_KEY": "openai-key"}, clear=True),
                mock.patch("main.OpenAI", return_value=object()),
            ):
                result = main.run_agentic_loop(
                    prproj_path=None,
                    raw_feedback_path=None,
                    project_name=project_name,
                    assets_dir="assets",
                    output_base_dir=temp_dir,
                    provider="openai",
                    from_stage="plan",
                )

            self.assertEqual(prompts_path, result)
            mock_timeline.assert_not_called()
            mock_align.assert_not_called()
            mock_ref_frames.assert_not_called()
            mock_plan.assert_not_called()
            mock_gen_prompts.assert_called_once_with(
                plan_json_path=plan_path,
                project_name=project_name,
                assets_dir="assets",
                openai_client=mock.ANY,
                openai_model="gpt-5.4-mini",
                output_base_dir=temp_dir,
            )

            manifest_path = os.path.join(project_dir, "agent_run_manifest.json")
            with open(manifest_path, "r", encoding="utf-8") as file:
                manifest = json.load(file)
            self.assertEqual("prompts", manifest["final_output"])
            self.assertEqual("reused", manifest["stages"]["plan"]["status"])
            self.assertEqual("ran", manifest["stages"]["prompts"]["status"])

    @mock.patch("main.setup_project_workspace")
    @mock.patch("main.extract_timeline_from_project")
    @mock.patch("main.parse_and_align_feedback")
    @mock.patch("main.analyze_and_extract_referenced_frames")
    @mock.patch("main.plan_generation_workflow")
    @mock.patch("main.generate_video_prompts_from_plan")
    def test_run_agentic_loop_accepts_project_package(
        self,
        mock_gen_prompts,
        mock_plan,
        mock_ref_frames,
        mock_align,
        mock_timeline,
        mock_setup,
    ):
        with tempfile.TemporaryDirectory() as temp_dir:
            project_name = "test_project"
            project_dir = os.path.join(temp_dir, "assets", project_name)
            prproj_path = os.path.join(project_dir, "project.prproj")
            timeline_path = os.path.join(temp_dir, "data", project_name, "timeline.json")
            mock_setup.return_value = (project_dir, prproj_path)
            mock_timeline.return_value = timeline_path

            with (
                mock.patch.dict(os.environ, {"OPENAI_API_KEY": "openai-key"}, clear=True),
                mock.patch("main.OpenAI", return_value=object()),
            ):
                result = main.run_agentic_loop(
                    prproj_path=None,
                    raw_feedback_path=None,
                    project_name=project_name,
                    assets_dir=os.path.join(temp_dir, "assets"),
                    output_base_dir=os.path.join(temp_dir, "data"),
                    provider="openai",
                    project_package_path="project.zip",
                    stage="timeline",
                )

            self.assertEqual(timeline_path, result)
            mock_setup.assert_called_once_with(
                zip_path="project.zip",
                project_name=project_name,
                assets_dir=os.path.join(temp_dir, "assets"),
            )
            mock_timeline.assert_called_once_with(
                prproj_path=prproj_path,
                project_name=project_name,
                output_base_dir=os.path.join(temp_dir, "data"),
            )
            mock_align.assert_not_called()
            mock_ref_frames.assert_not_called()
            mock_plan.assert_not_called()
            mock_gen_prompts.assert_not_called()


if __name__ == "__main__":
    unittest.main()
