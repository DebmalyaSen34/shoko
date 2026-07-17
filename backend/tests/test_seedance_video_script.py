import json
import os
import tempfile
import unittest
from unittest import mock

from scripts.generate_seedance_video import (
    build_seedance_content,
    build_segmind_payload,
    clamp_duration,
    load_prompt_item,
)


class SeedanceVideoScriptTests(unittest.TestCase):
    def test_load_prompt_item_selects_successful_video_item(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            prompt_path = os.path.join(temp_dir, "prompts.json")
            with open(prompt_path, "w", encoding="utf-8") as file:
                json.dump(
                    [
                        {"category": "audio", "status": "success"},
                        {
                            "category": "video",
                            "status": "success",
                            "video_model_prompt": "make this video",
                        },
                    ],
                    file,
                )

            item = load_prompt_item(prompt_path, index=0)

        self.assertEqual("make this video", item["video_model_prompt"])

    def test_build_content_uses_local_initial_frame_data_url(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = os.path.join(temp_dir, "initial.png")
            with open(image_path, "wb") as file:
                file.write(b"png-bytes")

            content = build_seedance_content(
                {
                    "video_model_prompt": "playful hall entrance",
                    "initial_frame_image_path": image_path,
                },
                use_local_initial_frame=True,
                initial_image_url=None,
            )

        self.assertEqual(
            {"type": "text", "text": "playful hall entrance"},
            content[0],
        )
        self.assertEqual("image_url", content[1]["type"])
        self.assertEqual(image_path, content[1]["image_url"]["url"])
        self.assertEqual("first_frame", content[1].get("role"))

    def test_build_content_with_first_frame_only(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = os.path.join(temp_dir, "initial.png")
            with open(image_path, "wb") as file:
                file.write(b"png-bytes")

            asset_path_1 = os.path.join(temp_dir, "character_sheet.png")
            with open(asset_path_1, "wb") as file:
                file.write(b"asset1-bytes")

            clip_frame_1 = os.path.join(temp_dir, "frame1.png")
            with open(clip_frame_1, "wb") as file:
                file.write(b"frame1-bytes")

            content = build_seedance_content(
                {
                    "video_model_prompt": "boy @image1 runs like @video1",
                    "initial_frame_image_path": image_path,
                    "selected_assets": [asset_path_1],
                    "clip_frame_paths": [clip_frame_1],
                },
                use_local_initial_frame=True,
                initial_image_url=None,
            )

        self.assertEqual(
            {"type": "text", "text": "boy image 1 runs like image 2"},
            content[0],
        )
        self.assertEqual("first_frame", content[1].get("role"))
        self.assertEqual("reference_image", content[2].get("role"))
        self.assertEqual("reference_image", content[3].get("role"))
        self.assertEqual(4, len(content))  # text + first frame + asset + clip frame

    def test_build_content_with_assets_and_clip_frames_mapping_without_first_frame(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            asset_path_1 = os.path.join(temp_dir, "character_sheet.png")
            with open(asset_path_1, "wb") as file:
                file.write(b"asset1-bytes")

            clip_frame_1 = os.path.join(temp_dir, "frame1.png")
            with open(clip_frame_1, "wb") as file:
                file.write(b"frame1-bytes")

            content = build_seedance_content(
                {
                    "video_model_prompt": "boy @image1 runs like @video1",
                    "initial_frame_image_path": "",
                    "selected_assets": [asset_path_1],
                    "clip_frame_paths": [clip_frame_1],
                },
                use_local_initial_frame=False,
                initial_image_url=None,
            )

        self.assertEqual(
            {"type": "text", "text": "boy image 1 runs like image 2"},
            content[0],
        )
        self.assertEqual("reference_image", content[1].get("role"))
        self.assertEqual("reference_image", content[2].get("role"))
        self.assertEqual(3, len(content))  # text + 2 reference images (asset1, frame1)

    def test_build_content_only_reference_video(self):
        content = build_seedance_content(
            {
                "video_model_prompt": "boy @image1 runs like @video1",
                "selected_assets": ["assets/project-red-and-green/01_characters/vir-sheet-v2.png"],
            },
            use_local_initial_frame=False,
            initial_image_url=None,
            only_reference_video=True,
            reference_video_url="https://example.com/video.mp4",
        )

        self.assertEqual(2, len(content))
        self.assertEqual(
            {"type": "text", "text": "boy the character sheet runs like video 1"},
            content[0],
        )
        self.assertEqual(
            {
                "type": "video_url",
                "video_url": {"url": "https://example.com/video.mp4"},
                "role": "reference_video",
            },
            content[1],
        )

    def test_build_content_with_image_urls_in_clip_frame_paths(self):
        from scripts.generate_seedance_video import build_seedance_content
        content = build_seedance_content(
            {
                "video_model_prompt": "movement from @video1",
                "initial_frame_image_path": None,
                "clip_frame_paths": [
                    "https://example.com/frame1.jpg",
                    "https://example.com/frame2.jpg",
                ]
            },
            use_local_initial_frame=False,
            initial_image_url=None,
        )

        self.assertEqual(3, len(content))
        self.assertEqual(
            {"type": "text", "text": "movement from image 1, image 2"},
            content[0],
        )
        self.assertEqual(
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/frame1.jpg"},
                "role": "reference_image",
            },
            content[1],
        )
        self.assertEqual(
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/frame2.jpg"},
                "role": "reference_image",
            },
            content[2],
        )

    def test_build_content_with_referenced_frame_paths_mapping(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            ref_path = os.path.join(temp_dir, "vir_smile.jpg")
            with open(ref_path, "wb") as file:
                file.write(b"ref-bytes")

            content = build_seedance_content(
                {
                    "video_model_prompt": "hard cut to @ref1 for Vir smile",
                    "initial_frame_image_path": "",
                    "referenced_frame_paths": [ref_path],
                    "referenced_frame_labels": ["@ref1"],
                },
                use_local_initial_frame=False,
                initial_image_url=None,
            )

        self.assertEqual(
            {"type": "text", "text": "hard cut to image 1 for Vir smile"},
            content[0],
        )
        self.assertEqual("reference_image", content[1].get("role"))
        self.assertEqual(ref_path, content[1]["image_url"]["url"])

    def test_duration_is_clamped_to_segmind_range(self):
        self.assertEqual(8, clamp_duration(8))
        self.assertEqual(4, clamp_duration(1))
        self.assertEqual(4, clamp_duration(4))
        self.assertEqual(15, clamp_duration(20))

    def test_build_segmind_payload_uses_api_reference_fields(self):
        payload = build_segmind_payload(
            item={
                "video_model_prompt": "use @image1 and @video1",
                "selected_assets": ["https://example.com/character.jpg"],
                "clip_frame_paths": ["https://example.com/frame.jpg"],
                "audio_url": "https://example.com/audio.mp3",
                "duration": 12,
                "ratio": "9:16",
                "generate_audio": True,
            },
            api_key="test-key",
            cache=None,
            upload_assets=False,
        )

        self.assertEqual("use image 1 and image 2", payload["prompt"])
        self.assertEqual(["https://example.com/character.jpg", "https://example.com/frame.jpg"], payload["reference_images"])
        self.assertEqual(["https://example.com/audio.mp3"], payload["reference_audios"])
        self.assertEqual(12, payload["duration"])
        self.assertEqual("9:16", payload["aspect_ratio"])
        self.assertTrue(payload["generate_audio"])


if __name__ == "__main__":
    unittest.main()
