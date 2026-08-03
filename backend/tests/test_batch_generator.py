import os
import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from src.generator import (
    OpenAIFileReference,
    _openai_file_reference,
    generate_single_video_prompt,
    generate_video_prompts_batch,
)
from src.generator.orchestrator import _refine_prompt
from src.generator.prompts import format_prompt_lessons
from src.selector import scan_visual_reference_assets
from config.settings import OPENAI_IMAGE_MODEL


class BatchConfigurationTests(unittest.TestCase):
    def test_batch_size_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "batch_size must be positive"):
            generate_video_prompts_batch(
                client=object(),
                clusters=[],
                reference_assets=[],
                assets_dir="assets",
                batch_size=0,
            )

    def test_visual_reference_scan_excludes_clips_and_audio(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            character_dir = os.path.join(temp_dir, "01_characters")
            location_dir = os.path.join(temp_dir, "03_locations")
            clip_dir = os.path.join(temp_dir, "06_clips")
            audio_dir = os.path.join(temp_dir, "04_audio")
            for directory in (character_dir, location_dir, clip_dir, audio_dir):
                os.makedirs(directory)

            character_path = os.path.join(character_dir, "character.png")
            location_path = os.path.join(location_dir, "location.png")
            for path in (
                character_path,
                location_path,
                os.path.join(clip_dir, "clip.mp4"),
                os.path.join(audio_dir, "audio.wav"),
                os.path.join(character_dir, ".DS_Store"),
            ):
                with open(path, "w", encoding="utf-8") as file:
                    file.write("test")

            references = scan_visual_reference_assets(temp_dir)

        self.assertEqual(sorted([character_path, location_path]), references)


class BatchGeneratorTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.assets_dir = self.temp_dir.name
        self.raw_dir = os.path.join(self.assets_dir, "06_clips", "_raw")
        os.makedirs(self.raw_dir)
        self.reference_assets = []
        for filename in ("character.png", "location.png"):
            path = os.path.join(self.assets_dir, filename)
            with open(path, "w", encoding="utf-8") as file:
                file.write("reference")
            self.reference_assets.append(path)

        self.extract_frames_patcher = mock.patch(
            "src.generator.media._extract_video_frames",
            side_effect=lambda path, out_dir, duration_s=0.0: [
                os.path.join(out_dir, "frame_001.jpg"),
                os.path.join(out_dir, "frame_002.jpg"),
                os.path.join(out_dir, "frame_003.jpg"),
            ]
        )
        self.mock_extract = self.extract_frames_patcher.start()

        def mock_quality_check(provider, client, model, prompt, feedback_items, selected_assets, has_clip=True):
            word_count = len((prompt or "").split())
            passed = word_count >= 150
            suggestions = []
            if not passed:
                suggestions.append(f"Detailed prompt contains {word_count} words; the target minimum is 150 words.")
            return {
                "passed": passed,
                "feedback_adherence": "Adheres perfectly.",
                "clothing_consistency": "Consistent.",
                "forbidden_terms_found": [],
                "vague_feedback_resolution": "Resolved.",
                "suggestions": suggestions
            }

        self.validator_patcher = mock.patch(
            "src.generator.orchestrator.run_quality_check",
            side_effect=mock_quality_check
        )
        self.mock_validator = self.validator_patcher.start()

    def tearDown(self):
        self.extract_frames_patcher.stop()
        self.validator_patcher.stop()
        self.temp_dir.cleanup()

    def make_clusters(self, count):
        clusters = []
        for cluster_id in range(count):
            filename = f"clip-{cluster_id}.mp4"
            with open(os.path.join(self.raw_dir, filename), "w", encoding="utf-8") as file:
                file.write("video")
            clusters.append(
                {
                    "category": "video",
                    "clip_occurrence": cluster_id,
                    "frame_size": "1080x1920",
                    "matched_clip": {
                        "clip": filename,
                        "start_tc": f"00:00:{cluster_id:02d}:00",
                        "end_tc": f"00:00:{cluster_id + 1:02d}:00",
                        "duration_s": 1.0,
                    },
                    "feedback_items": [
                        {"timestamp": f"00:{cluster_id:02d}", "remark": f"change {cluster_id}"}
                    ],
                }
            )
        return clusters

    @staticmethod
    def response_for_ids(cluster_ids):
        return SimpleNamespace(
            text=json.dumps(
                {
                    "results": [
                        {
                            "cluster_id": cluster_id,
                            "selected_assets": ["character.png"],
                            "prompt_format": "plain_text",
                            "reference_legend": "@image1 — character.png\n@video1 — original clip",
                            "english_prompt": (
                                f"prompt {cluster_id} " + " ".join(["specific"] * 159)
                            ),
                            "explanation": f"explanation {cluster_id}",
                        }
                        for cluster_id in reversed(cluster_ids)
                    ]
                }
            )
        )

    @staticmethod
    def initial_frame_response_for_ids(cluster_ids):
        return SimpleNamespace(
            text=json.dumps(
                {
                    "results": [
                        {
                            "cluster_id": cluster_id,
                            "initial_frame_prompt": (
                                f"initial frame {cluster_id}: vertical cinematic frame "
                                "anchored to the supplied character sheet, location sheet, "
                                "and original clip continuity"
                            ),
                            "explanation": f"initial explanation {cluster_id}",
                        }
                        for cluster_id in cluster_ids
                    ]
                }
            )
        )

    @staticmethod
    def selection_response_for_ids(cluster_ids, selected_assets):
        return SimpleNamespace(
            text=json.dumps(
                {
                    "results": [
                        {
                            "cluster_id": cluster_id,
                            "selected_assets": selected_assets,
                            "reasoning": "Selected minimal matching references.",
                        }
                        for cluster_id in cluster_ids
                    ]
                }
            )
        )

    def test_generates_initial_frame_prompts_before_video_prompts(self):
        clusters = self.make_clusters(1)
        client = mock.MagicMock()
        client.models.generate_content.side_effect = [
            self.selection_response_for_ids(range(1), ["character.png"]),
            self.initial_frame_response_for_ids(range(1)),
            self.response_for_ids(range(1)),
        ]

        with mock.patch(
            "src.generator.upload.upload_file_and_wait",
            side_effect=lambda _client, path: SimpleNamespace(name=f"uploaded-{path}"),
        ):
            results = generate_video_prompts_batch(
                client=client,
                clusters=clusters,
                reference_assets=self.reference_assets,
                assets_dir=self.assets_dir,
            )

        self.assertEqual(3, client.models.generate_content.call_count)
        self.assertIn("initial frame 0", results[0]["initial_frame_prompt"])

        first_call_text = "\n".join(
            item
            for item in client.models.generate_content.call_args_list[1].kwargs["contents"]
            if isinstance(item, str)
        )
        second_call_text = "\n".join(
            item
            for item in client.models.generate_content.call_args_list[2].kwargs["contents"]
            if isinstance(item, str)
        )
        self.assertIn("Generate an initial frame prompt", first_call_text)
        self.assertIn("INITIAL_FRAME_PROMPT", second_call_text)
        self.assertIn("initial frame 0", second_call_text)

    def test_format_prompt_lessons_caps_numbered_block_at_three(self):
        block = format_prompt_lessons(
            [
                {"id": f"lesson-{index}", "lesson": f"Lesson text {index}"}
                for index in range(4)
            ]
        )

        self.assertIn("RELEVANT LEARNED LESSONS FROM PRIOR FEEDBACK:", block)
        self.assertIn("1. Lesson text 0", block)
        self.assertIn("3. Lesson text 2", block)
        self.assertNotIn("Lesson text 3", block)

    def test_batch_generation_injects_prompt_lessons_and_persists_metadata(self):
        clusters = self.make_clusters(1)
        client = mock.MagicMock()
        client.models.generate_content.side_effect = [
            self.selection_response_for_ids(range(1), ["character.png"]),
            self.response_for_ids(range(1)),
        ]
        lessons = [
            {
                "id": "lesson-continuity",
                "scope": "project",
                "category": "continuity_error",
                "lesson": "Preserve wardrobe and location explicitly when feedback mentions continuity.",
            },
            {
                "id": "lesson-motion",
                "scope": "clip",
                "category": "too_vague",
                "lesson": "Convert vague speed notes into concrete motion language.",
            },
        ]

        with mock.patch(
            "src.generator.upload.upload_file_and_wait",
            side_effect=lambda _client, path: SimpleNamespace(name=f"uploaded-{path}"),
        ):
            results = generate_video_prompts_batch(
                client=client,
                clusters=clusters,
                reference_assets=self.reference_assets,
                assets_dir=self.assets_dir,
                generate_initial_frame=False,
                run_validator=False,
                prompt_lessons_by_cluster={0: lessons},
            )

        final_text = "\n".join(
            item
            for item in client.models.generate_content.call_args_list[1].kwargs["contents"]
            if isinstance(item, str)
        )
        self.assertIn("RELEVANT LEARNED LESSONS FROM PRIOR FEEDBACK:", final_text)
        self.assertIn("Preserve wardrobe and location explicitly", final_text)
        self.assertIn("REFERENCE_LEGEND_TO_USE", final_text)
        self.assertLess(
            final_text.index("RELEVANT LEARNED LESSONS FROM PRIOR FEEDBACK:"),
            final_text.index("REFERENCE_LEGEND_TO_USE"),
        )
        self.assertEqual("lesson-continuity", results[0]["applied_prompt_lessons"][0]["id"])
        self.assertEqual("project", results[0]["applied_prompt_lessons"][0]["scope"])

    def test_batch_generation_without_lessons_omits_lesson_metadata(self):
        clusters = self.make_clusters(1)
        client = mock.MagicMock()
        client.models.generate_content.side_effect = [
            self.selection_response_for_ids(range(1), ["character.png"]),
            self.response_for_ids(range(1)),
        ]

        with mock.patch(
            "src.generator.upload.upload_file_and_wait",
            side_effect=lambda _client, path: SimpleNamespace(name=f"uploaded-{path}"),
        ):
            results = generate_video_prompts_batch(
                client=client,
                clusters=clusters,
                reference_assets=self.reference_assets,
                assets_dir=self.assets_dir,
                generate_initial_frame=False,
                run_validator=False,
            )

        final_text = "\n".join(
            item
            for item in client.models.generate_content.call_args_list[1].kwargs["contents"]
            if isinstance(item, str)
        )
        self.assertNotIn("RELEVANT LEARNED LESSONS FROM PRIOR FEEDBACK", final_text)
        self.assertNotIn("applied_prompt_lessons", results[0])

    def test_refine_prompt_includes_lessons_to_avoid_repeated_mistakes(self):
        with mock.patch(
            "src.generator.orchestrator.generate_structured",
            return_value={"english_prompt": "refined prompt", "refinement_explanation": "ok"},
        ) as generated:
            refined = _refine_prompt(
                provider="gemini",
                client=mock.MagicMock(),
                model="gemini-test",
                draft_prompt="draft prompt",
                feedback_items=[{"remark": "keep continuity"}],
                suggestions=["Add concrete wardrobe details."],
                lessons=[
                    {
                        "id": "lesson-1",
                        "lesson": "Do not change wardrobe when continuity is requested.",
                        "category": "continuity_error",
                        "scope": "project",
                    }
                ],
            )

        self.assertEqual("refined prompt", refined)
        refiner_text = generated.call_args.kwargs["contents"][0]
        self.assertIn("KNOWN PRIOR MISTAKES TO AVOID:", refiner_text)
        self.assertIn("Do not change wardrobe", refiner_text)

    def test_batch_generation_merges_learning_eval_and_refines_from_its_suggestions(self):
        clusters = self.make_clusters(1)
        client = mock.MagicMock()
        client.models.generate_content.side_effect = [
            self.selection_response_for_ids(range(1), ["character.png"]),
            self.response_for_ids(range(1)),
        ]
        learning_fail = {
            "passed": False,
            "score": 0.55,
            "failed_cases": ["case-1"],
            "case_results": [],
            "suggestions": ["Prompt does not explicitly preserve wardrobe continuity."],
        }
        learning_pass = {
            "passed": True,
            "score": 0.95,
            "failed_cases": [],
            "case_results": [],
            "suggestions": [],
        }

        with mock.patch(
            "src.generator.upload.upload_file_and_wait",
            side_effect=lambda _client, path: SimpleNamespace(name=f"uploaded-{path}"),
        ), mock.patch(
            "src.generator.orchestrator.run_learning_eval",
            side_effect=[learning_fail, learning_pass],
        ), mock.patch(
            "src.generator.orchestrator._refine_prompt",
            return_value="refined prompt " + " ".join(["specific"] * 160),
        ) as refine:
            results = generate_video_prompts_batch(
                client=client,
                clusters=clusters,
                reference_assets=self.reference_assets,
                assets_dir=self.assets_dir,
                generate_initial_frame=False,
                prompt_eval_cases_by_cluster={
                    0: [
                        {
                            "id": "case-1",
                            "enabled": True,
                            "expected_behavior": ["Preserve wardrobe continuity."],
                            "input": {"selected_assets": ["character.png"]},
                        }
                    ]
                },
            )

        refine.assert_called_once()
        self.assertIn(
            "Prompt does not explicitly preserve wardrobe continuity.",
            refine.call_args.kwargs["suggestions"],
        )
        self.assertTrue(results[0]["quality_report"]["learning_eval"]["passed"])

    def test_batch_generation_uploads_only_assets_selected_for_the_cluster(self):
        clusters = self.make_clusters(1)
        selected_character = self.reference_assets[0]
        selected_location = self.reference_assets[1]
        unused_assets = []
        for filename in ("unused-avni.png", "unused-rian.png"):
            path = os.path.join(self.assets_dir, filename)
            with open(path, "w", encoding="utf-8") as file:
                file.write("unused reference")
            unused_assets.append(path)

        all_assets = [
            selected_character,
            *unused_assets,
            selected_location,
        ]
        client = mock.MagicMock()
        client.models.generate_content.side_effect = [
            SimpleNamespace(
                text=json.dumps(
                    {
                        "results": [
                            {
                                "cluster_id": 0,
                                "selected_assets": [
                                    selected_character,
                                    selected_location,
                                ],
                                "reasoning": "Vir and the home interior are the only needed references.",
                            }
                        ]
                    }
                )
            ),
            self.initial_frame_response_for_ids(range(1)),
            self.response_for_ids(range(1)),
        ]

        with mock.patch(
            "src.generator.upload.upload_file_and_wait",
            side_effect=lambda _client, path: SimpleNamespace(name=f"uploaded-{path}"),
        ) as upload:
            results = generate_video_prompts_batch(
                client=client,
                clusters=clusters,
                reference_assets=all_assets,
                assets_dir=self.assets_dir,
                batch_size=1,
            )

        uploaded_paths = [os.path.normpath(call.args[1]) for call in upload.call_args_list]
        self.assertCountEqual(
            [os.path.normpath(path) for path in [
                selected_character,
                selected_location,
                "data/output/video_frames/clip_0_clip-0/frame_001.jpg",
                "data/output/video_frames/clip_0_clip-0/frame_002.jpg",
                "data/output/video_frames/clip_0_clip-0/frame_003.jpg",
            ]],
            uploaded_paths,
        )
        self.assertEqual([selected_character, selected_location], results[0]["selected_assets"])
        self.assertIn(os.path.basename(selected_character), results[0]["reference_legend"])
        self.assertIn(os.path.basename(selected_location), results[0]["reference_legend"])
        self.assertNotIn(os.path.basename(unused_assets[0]), results[0]["reference_legend"])

        for call_index in (1, 2):
            contents = client.models.generate_content.call_args_list[call_index].kwargs["contents"]
            text_contents = "\n".join(item for item in contents if isinstance(item, str))
            self.assertIn(selected_character, text_contents)
            self.assertIn(selected_location, text_contents)
            for unused_asset in unused_assets:
                self.assertNotIn(unused_asset, text_contents)

    def test_openai_provider_sends_typed_responses_content(self):
        clusters = self.make_clusters(1)
        client = mock.MagicMock()
        client.images.generate.return_value = SimpleNamespace(
            data=[SimpleNamespace(b64_json="Z2VuZXJhdGVkLWZyYW1l")]
        )
        client.responses.parse.side_effect = [
            SimpleNamespace(
                output_parsed=SimpleNamespace(
                    model_dump=lambda: json.loads(
                        self.selection_response_for_ids(range(1), ["character.png"]).text
                    )
                )
            ),
            SimpleNamespace(
                output_parsed=SimpleNamespace(
                    model_dump=lambda: json.loads(
                        self.initial_frame_response_for_ids(range(1)).text
                    )
                )
            ),
            SimpleNamespace(
                output_parsed=SimpleNamespace(
                    model_dump=lambda: json.loads(self.response_for_ids(range(1)).text)
                )
            ),
        ]

        def upload_reference(_client, path, provider=None):
            if path.endswith(".mp4"):
                self.fail("OpenAI should use extracted frames instead of raw mp4 uploads")
            return OpenAIFileReference(
                content_block={
                    "type": "input_image",
                    "image_url": "data:image/png;base64,cmVm",
                    "detail": "auto",
                }
            )

        def extract_frames(path, output_dir, duration_s, frame_count=3):
            self.assertTrue(path.endswith(".mp4"))
            frame_paths = []
            for index in range(frame_count):
                frame_path = os.path.join(output_dir, f"frame_{index + 1:03d}.jpg")
                os.makedirs(os.path.dirname(frame_path), exist_ok=True)
                with open(frame_path, "wb") as file:
                    file.write(f"frame-{index + 1}".encode("ascii"))
                frame_paths.append(frame_path)
            return frame_paths

        with mock.patch(
            "src.generator.upload.upload_file_and_wait",
            side_effect=upload_reference,
        ), mock.patch(
            "src.generator.media._extract_video_frames",
            side_effect=extract_frames,
        ) as extract, mock.patch.dict(os.environ, {"OPENAI_MODEL": "gpt-4.1-mini"}):
            results = generate_video_prompts_batch(
                client=client,
                clusters=clusters,
                reference_assets=self.reference_assets,
                assets_dir=self.assets_dir,
                provider="openai",
                initial_frames_dir=self.temp_dir.name,
                video_frames_dir=os.path.join(self.temp_dir.name, "video_frames"),
            )

        self.assertIn("prompt 0", results[0]["video_model_prompt"])
        self.assertTrue(os.path.exists(results[0]["initial_frame_image_path"]))
        with open(results[0]["initial_frame_image_path"], "rb") as file:
            self.assertEqual(b"generated-frame", file.read())
        client.images.generate.assert_called_once()
        self.assertEqual(OPENAI_IMAGE_MODEL, client.images.generate.call_args.kwargs["model"])
        self.assertNotIn("response_format", client.images.generate.call_args.kwargs)
        self.assertEqual("png", client.images.generate.call_args.kwargs["output_format"])
        self.assertIn(
            "initial frame 0",
            client.images.generate.call_args.kwargs["prompt"],
        )
        self.assertEqual(3, client.responses.parse.call_count)
        self.assertEqual(0, client.models.generate_content.call_count)
        self.assertEqual("gpt-4.1-mini", client.responses.parse.call_args_list[0].kwargs["model"])
        extract.assert_called_once()
        self.assertEqual(
            [
                os.path.join(
                    self.temp_dir.name,
                    "video_frames",
                    "clip_0_clip-0",
                    f"frame_{index:03d}.jpg",
                )
                for index in range(1, 4)
            ],
            results[0]["clip_frame_paths"],
        )
        selection_input = client.responses.parse.call_args_list[0].kwargs["input"]
        selection_content_types = [block["type"] for block in selection_input[0]["content"]]
        first_input = client.responses.parse.call_args_list[1].kwargs["input"]
        first_content_types = [block["type"] for block in first_input[0]["content"]]
        final_input = client.responses.parse.call_args_list[2].kwargs["input"]
        final_blocks = final_input[0]["content"]
        final_text = "\n".join(
            block["text"] for block in final_blocks if block["type"] == "input_text"
        )
        final_image_urls = [
            block["image_url"] for block in final_blocks if block["type"] == "input_image"
        ]
        self.assertEqual({"input_text"}, set(selection_content_types))
        self.assertIn("input_text", first_content_types)
        self.assertIn("input_image", first_content_types)
        self.assertNotIn("input_file", first_content_types)
        self.assertEqual(4, first_content_types.count("input_image"))
        for block in first_input[0]["content"] + final_blocks:
            if block["type"] == "input_file":
                self.assertNotEqual(
                    {"file_id", "filename"},
                    {"file_id", "filename"} & block.keys(),
                )
        self.assertIn("GENERATED_INITIAL_FRAME_IMAGE", final_text)
        self.assertTrue(
            any("Z2VuZXJhdGVkLWZyYW1l" in image_url for image_url in final_image_urls)
        )
        self.assertTrue(any("ZnJhbWUtMQ==" in image_url for image_url in final_image_urls))
        client.files.delete.assert_not_called()

    def test_openai_uploaded_file_reference_uses_file_id_without_filename(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            temp_file.write(b"video")
            temp_file.flush()
            temp_file_path = temp_file.name

        try:
            client = mock.MagicMock()
            client.files.create.return_value = SimpleNamespace(id="file_uploaded")

            ref = _openai_file_reference(client, temp_file_path)
        finally:
            os.unlink(temp_file_path)

        self.assertEqual(
            {
                "type": "input_file",
                "file_id": "file_uploaded",
            },
            ref.content_block,
        )

    def test_single_video_prompt_uses_one_cluster_without_extra_batches(self):
        clusters = self.make_clusters(3)
        client = mock.MagicMock()
        client.models.generate_content.side_effect = [
            self.selection_response_for_ids(range(1), ["character.png"]),
            self.initial_frame_response_for_ids(range(1)),
            self.response_for_ids(range(1)),
        ]

        with mock.patch(
            "src.generator.upload.upload_file_and_wait",
            side_effect=lambda _client, path: SimpleNamespace(name=f"uploaded-{path}"),
        ) as upload:
            result = generate_single_video_prompt(
                client=client,
                cluster=clusters[1],
                reference_assets=self.reference_assets,
                assets_dir=self.assets_dir,
            )

        self.assertEqual(3, client.models.generate_content.call_count)
        self.assertEqual(4, upload.call_count)
        self.assertIn("prompt 0", result["video_model_prompt"])
        self.assertIn("initial frame 0", result["initial_frame_prompt"])

    def test_thirteen_clusters_use_three_calls_and_map_reordered_results(self):
        clusters = self.make_clusters(13)
        client = mock.MagicMock()
        client.models.generate_content.side_effect = [
            self.selection_response_for_ids(range(0, 5), ["character.png"]),
            self.initial_frame_response_for_ids(range(0, 5)),
            self.response_for_ids(range(0, 5)),
            self.selection_response_for_ids(range(5, 10), ["character.png"]),
            self.initial_frame_response_for_ids(range(5, 10)),
            self.response_for_ids(range(5, 10)),
            self.selection_response_for_ids(range(10, 13), ["character.png"]),
            self.initial_frame_response_for_ids(range(10, 13)),
            self.response_for_ids(range(10, 13)),
        ]

        with mock.patch(
            "src.generator.upload.upload_file_and_wait",
            side_effect=lambda _client, path: SimpleNamespace(name=f"uploaded-{path}"),
        ) as upload:
            results = generate_video_prompts_batch(
                client=client,
                clusters=clusters,
                reference_assets=self.reference_assets,
                assets_dir=self.assets_dir,
                batch_size=5,
            )

        self.assertEqual(9, client.models.generate_content.call_count)
        self.assertEqual(13, len(results))
        self.assertIn("prompt 0", results[0]["video_model_prompt"])
        self.assertIn("initial frame 0", results[0]["initial_frame_prompt"])
        self.assertIn("prompt 12", results[12]["video_model_prompt"])
        self.assertEqual("plain_text", results[0]["prompt_format"])
        self.assertIn("original clip reference images", results[0]["reference_legend"])
        self.assertEqual("success", results[0]["status"])
        self.assertIsNone(results[0]["quality_warning"])
        self.assertEqual(40, upload.call_count)
        self.assertEqual(40, client.files.delete.call_count)
        first_contents = client.models.generate_content.call_args_list[2].kwargs["contents"]
        text_contents = "\n".join(item for item in first_contents if isinstance(item, str))
        self.assertIn("FRAME_SIZE: 1080x1920", text_contents)
        self.assertIn("ASPECT_RATIO: 9:16", text_contents)
        self.assertNotIn("TIMELINE_SEGMENT", text_contents)
        default_instruction = client.models.generate_content.call_args_list[2].kwargs[
            "config"
        ].system_instruction
        self.assertIn("# Seedance 2.0 — Universal Director", default_instruction)
        self.assertIn("PROJECT OVERRIDES", default_instruction)

    def test_direct_skill_file_is_sent_before_project_overrides(self):
        clusters = self.make_clusters(1)
        skill_marker = "UNIQUE FULL SKILL CONTENT MARKER"
        client = mock.MagicMock()
        client.models.generate_content.side_effect = [
            self.selection_response_for_ids(range(1), ["character.png"]),
            self.initial_frame_response_for_ids(range(1)),
            self.response_for_ids(range(1)),
        ]

        with mock.patch(
            "src.generator.upload.upload_file_and_wait",
            side_effect=lambda _client, path: SimpleNamespace(name=f"uploaded-{path}"),
        ):
            generate_video_prompts_batch(
                client,
                clusters,
                self.reference_assets,
                self.assets_dir,
                prompt_skill_text=skill_marker,
            )

        instruction = client.models.generate_content.call_args_list[2].kwargs[
            "config"
        ].system_instruction
        self.assertIn(skill_marker, instruction)
        self.assertIn("English output for all prompt instructions", instruction)
        self.assertIn("Never use timeline prompting", instruction)
        self.assertLess(instruction.index(skill_marker), instruction.index("PROJECT OVERRIDES"))

    def test_short_skill_prompt_is_retained_with_warning(self):
        clusters = self.make_clusters(1)
        client = mock.MagicMock()
        client.models.generate_content.side_effect = [
            self.selection_response_for_ids(range(1), ["character.png"]),
            self.initial_frame_response_for_ids(range(1)),
            SimpleNamespace(
                text=json.dumps(
                    {
                        "results": [
                            {
                                "cluster_id": 0,
                                "selected_assets": [],
                                "prompt_format": "plain_text",
                                "reference_legend": "@video1 — original clip",
                                "english_prompt": "short usable prompt",
                                "explanation": "short",
                            }
                        ]
                    }
                )
            ),
        ]

        with mock.patch(
            "src.generator.upload.upload_file_and_wait",
            side_effect=lambda _client, path: SimpleNamespace(name=f"uploaded-{path}"),
        ):
            results = generate_video_prompts_batch(
                client, clusters, self.reference_assets, self.assets_dir
            )

        self.assertEqual("short usable prompt", results[0]["video_model_prompt"])
        self.assertEqual("warning", results[0]["status"])
        self.assertIn("150 words", results[0]["quality_warning"])

    def test_missing_raw_clip_does_not_block_the_batch(self):
        clusters = self.make_clusters(2)
        os.remove(os.path.join(self.raw_dir, "clip-1.mp4"))
        client = mock.MagicMock()
        client.models.generate_content.side_effect = [
            self.selection_response_for_ids(range(2), ["character.png"]),
            self.initial_frame_response_for_ids(range(2)),
            self.response_for_ids(range(2)),
        ]

        with mock.patch(
            "src.generator.upload.upload_file_and_wait",
            side_effect=lambda _client, path: SimpleNamespace(name=f"uploaded-{path}"),
        ):
            results = generate_video_prompts_batch(
                client, clusters, self.reference_assets, self.assets_dir
            )

        self.assertEqual(2, len(results))
        self.assertEqual(3, client.models.generate_content.call_count)

    def test_missing_response_id_and_failed_batch_are_not_retried(self):
        clusters = self.make_clusters(7)
        client = mock.MagicMock()
        client.models.generate_content.side_effect = [
            self.selection_response_for_ids(range(0, 5), ["character.png"]),
            self.initial_frame_response_for_ids(range(0, 5)),
            self.response_for_ids(range(0, 4)),
            self.selection_response_for_ids(range(5, 7), ["character.png"]),
            self.initial_frame_response_for_ids(range(5, 7)),
            RuntimeError("quota failure"),
        ]

        with mock.patch(
            "src.generator.upload.upload_file_and_wait",
            side_effect=lambda _client, path: SimpleNamespace(name=f"uploaded-{path}"),
        ):
            results = generate_video_prompts_batch(
                client, clusters, self.reference_assets, self.assets_dir, batch_size=5
            )

        self.assertEqual(6, client.models.generate_content.call_count)
        self.assertEqual("failed", results[4]["status"])
        self.assertIn("omitted", results[4]["explanation"])
        self.assertEqual("failed", results[5]["status"])
        self.assertIn("quota failure", results[5]["explanation"])
        self.assertEqual("failed", results[6]["status"])

    def test_generate_video_prompts_without_initial_frame(self):
        clusters = self.make_clusters(1)
        client = mock.MagicMock()
        client.models.generate_content.side_effect = [
            self.selection_response_for_ids(range(1), ["character.png"]),
            self.response_for_ids(range(1)),
        ]

        with mock.patch(
            "src.generator.upload.upload_file_and_wait",
            side_effect=lambda _client, path: SimpleNamespace(name=f"uploaded-{path}"),
        ):
            results = generate_video_prompts_batch(
                client,
                clusters,
                self.reference_assets,
                self.assets_dir,
                batch_size=5,
                run_validator=False,
                generate_initial_frame=False,
            )

        self.assertEqual(2, client.models.generate_content.call_count)
        self.assertEqual("success", results[0]["status"])
        self.assertEqual("", results[0]["initial_frame_prompt"])
        self.assertEqual("", results[0]["initial_frame_image_path"])
        self.assertEqual("prompt 0 " + " ".join(["specific"] * 159), results[0]["video_model_prompt"])


if __name__ == "__main__":
    unittest.main()
