import json
import os
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest import mock

from scripts.generate_seedance_video import (
    SQLiteAssetUrlCache,
    SupabaseAssetUrlCache,
    attach_prepared_segmind_payload,
    build_seedance_content,
    build_segmind_payload,
    clamp_duration,
    create_asset_url_cache,
    create_seedance_task,
    load_prompt_item,
)


class SeedanceVideoScriptTests(unittest.TestCase):
    def test_sqlite_cache_creates_table_and_returns_none_for_miss(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "cache" / "asset_url_cache.sqlite3"
            cache = SQLiteAssetUrlCache(db_path)

            self.assertTrue(db_path.exists())
            self.assertIsNone(cache.get(source_hash="missing", provider="segmind", media_type="image"))

            with closing(sqlite3.connect(db_path)) as connection:
                with connection:
                    table_names = {
                        row[0]
                        for row in connection.execute(
                            "SELECT name FROM sqlite_master WHERE type = 'table'"
                        ).fetchall()
                    }
            self.assertIn("media_asset_urls", table_names)

    def test_sqlite_cache_upserts_and_retrieves_asset(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = SQLiteAssetUrlCache(Path(temp_dir) / "asset_url_cache.sqlite3")

            cache.upsert(
                source_hash="hash-1",
                provider="segmind",
                media_type="image",
                public_url="https://example.com/one.jpg",
                source_path="/tmp/one.jpg",
                metadata={"width": 1280},
            )
            first = cache.get(source_hash="hash-1", provider="segmind", media_type="image")

            self.assertIsNotNone(first)
            self.assertEqual("https://example.com/one.jpg", first.public_url)
            self.assertEqual("segmind", first.provider)
            self.assertEqual("image", first.media_type)
            self.assertEqual("hash-1", first.source_hash)

            cache.upsert(
                source_hash="hash-1",
                provider="segmind",
                media_type="image",
                public_url="https://example.com/two.jpg",
                source_path="/tmp/two.jpg",
                metadata={"width": 1920},
            )
            second = cache.get(source_hash="hash-1", provider="segmind", media_type="image")
            self.assertEqual("https://example.com/two.jpg", second.public_url)

            with closing(sqlite3.connect(cache.db_path)) as connection:
                with connection:
                    row = connection.execute(
                        """
                        SELECT metadata_json
                        FROM media_asset_urls
                        WHERE source_hash = ? AND provider = ? AND media_type = ?
                        """,
                        ("hash-1", "segmind", "image"),
                    ).fetchone()
            self.assertEqual({"width": 1920}, json.loads(row[0]))

    def test_cache_factory_defaults_to_sqlite_even_with_supabase_credentials(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "asset_url_cache.sqlite3"
            with mock.patch.dict(
                os.environ,
                {
                    "SUPABASE_URL": "https://example.supabase.co",
                    "SUPABASE_SERVICE_ROLE_KEY": "secret",
                },
                clear=True,
            ), mock.patch(
                "scripts.generate_seedance_video._default_sqlite_asset_cache_path",
                return_value=db_path,
            ):
                cache = create_asset_url_cache()

        self.assertIsInstance(cache, SQLiteAssetUrlCache)

    def test_cache_factory_selects_supabase_only_when_requested(self):
        with mock.patch.dict(
            os.environ,
            {
                "LOKA_ASSET_CACHE_BACKEND": "supabase",
                "SUPABASE_URL": "https://example.supabase.co",
                "SUPABASE_SERVICE_ROLE_KEY": "secret",
            },
            clear=True,
        ):
            cache = create_asset_url_cache()

        self.assertIsInstance(cache, SupabaseAssetUrlCache)

    def test_cache_factory_supabase_mode_requires_credentials(self):
        with mock.patch.dict(os.environ, {"LOKA_ASSET_CACHE_BACKEND": "supabase"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "requires SUPABASE_URL"):
                create_asset_url_cache()

    @mock.patch("scripts.generate_seedance_video.upload_data_url_to_segmind")
    def test_build_segmind_payload_caches_uploaded_image_url_in_sqlite(self, mock_upload):
        mock_upload.return_value = "https://segmind.example/uploaded.jpg"
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "frame.jpg"
            image_path.write_bytes(b"fake image bytes")
            cache = SQLiteAssetUrlCache(Path(temp_dir) / "asset_url_cache.sqlite3")

            first_payload = build_segmind_payload(
                item={
                    "video_model_prompt": "use @image1",
                    "selected_assets": [str(image_path)],
                },
                api_key="test-key",
                cache=cache,
            )
            second_payload = build_segmind_payload(
                item={
                    "video_model_prompt": "use @image1",
                    "selected_assets": [str(image_path)],
                },
                api_key="test-key",
                cache=cache,
            )

        self.assertEqual(["https://segmind.example/uploaded.jpg"], first_payload["reference_images"])
        self.assertEqual(["https://segmind.example/uploaded.jpg"], second_payload["reference_images"])
        mock_upload.assert_called_once()

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

        self.assertIn("first_frame_url is the first-frame continuity anchor", content[0]["text"])
        self.assertTrue(content[0]["text"].endswith("playful hall entrance"))
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

        self.assertIn("REFERENCE IMAGE MAP:", content[0]["text"])
        self.assertIn("image 1: character sheet", content[0]["text"])
        self.assertIn("image 2: original clip frame", content[0]["text"])
        self.assertTrue(content[0]["text"].endswith("boy image 1 runs like image 2"))
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

        self.assertIn("REFERENCE IMAGE MAP:", content[0]["text"])
        self.assertIn("image 1: character sheet", content[0]["text"])
        self.assertIn("image 2: original clip frame", content[0]["text"])
        self.assertTrue(content[0]["text"].endswith("boy image 1 runs like image 2"))
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
        self.assertIn("REFERENCE IMAGE MAP:", content[0]["text"])
        self.assertIn("image 1: original clip frame", content[0]["text"])
        self.assertIn("image 2: original clip frame", content[0]["text"])
        self.assertTrue(content[0]["text"].endswith("movement from image 1, image 2"))
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

        self.assertIn("REFERENCE IMAGE MAP:", content[0]["text"])
        self.assertIn("image 1: referenced cutaway/reaction frame", content[0]["text"])
        self.assertTrue(content[0]["text"].endswith("hard cut to image 1 for Vir smile"))
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

        self.assertIn("REFERENCE IMAGE MAP:", payload["prompt"])
        self.assertTrue(payload["prompt"].endswith("use image 1 and image 2"))
        self.assertEqual(["https://example.com/character.jpg", "https://example.com/frame.jpg"], payload["reference_images"])
        self.assertNotIn("reference_audios", payload)
        self.assertEqual(5, payload["duration"])
        self.assertEqual("9:16", payload["aspect_ratio"])
        self.assertFalse(payload["generate_audio"])

    def test_attach_prepared_segmind_payload_adds_ready_provider_fields(self):
        item = {
            "video_model_prompt": "use @image1 and @video1",
            "selected_assets": ["https://example.com/character.jpg"],
            "clip_frame_paths": ["https://example.com/frame.jpg"],
            "audio_url": "https://example.com/audio.mp3",
            "duration": 5,
            "ratio": "9:16",
            "generate_audio": True,
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            enriched = attach_prepared_segmind_payload(
                item,
                api_key="test-key",
                cache=SQLiteAssetUrlCache(Path(temp_dir) / "asset_url_cache.sqlite3"),
            )

        self.assertEqual("segmind", enriched["video_provider"])
        self.assertEqual("ready", enriched["segmind_payload_status"])
        self.assertIn("REFERENCE IMAGE MAP:", enriched["segmind_prompt"])
        self.assertTrue(enriched["segmind_prompt"].endswith("use image 1 and image 2"))
        self.assertEqual(enriched["segmind_prompt"], enriched["video_model_prompt"])
        self.assertEqual(5, enriched["duration"])
        self.assertFalse(enriched["generate_audio"])
        self.assertTrue(enriched["has_reference_audio"])
        self.assertEqual(
            ["https://example.com/character.jpg", "https://example.com/frame.jpg"],
            enriched["segmind_reference_images"],
        )
        self.assertEqual([], enriched["segmind_reference_audios"])
        self.assertEqual(enriched["segmind_payload"], build_segmind_payload(item=enriched, api_key="test-key", cache=None))

    def test_attach_prepared_segmind_payload_marks_missing_key_as_skipped(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            enriched = attach_prepared_segmind_payload(
                {"video_model_prompt": "prompt", "duration": 5},
                api_key=None,
                cache=None,
            )

        self.assertEqual("segmind", enriched["video_provider"])
        self.assertEqual("skipped", enriched["segmind_payload_status"])
        self.assertIn("SEGMIND_API_KEY", enriched["segmind_payload_error"])

    def test_attach_prepared_segmind_payload_preserves_local_reference_audio_without_upload(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = os.path.join(temp_dir, "trimmed_audio.mp3")
            with open(audio_path, "wb") as file:
                file.write(b"audio data")

            with mock.patch.dict(os.environ, {}, clear=True):
                enriched = attach_prepared_segmind_payload(
                    {
                        "video_model_prompt": "prompt",
                        "audio_reference_path": audio_path,
                        "duration": 5,
                    },
                    api_key=None,
                    cache=None,
                )

        self.assertEqual("skipped", enriched["segmind_payload_status"])
        self.assertTrue(enriched["has_reference_audio"])
        self.assertEqual([], enriched["segmind_reference_audios"])

    def test_build_segmind_payload_normalizes_stale_prepared_payload(self):
        payload = build_segmind_payload(
            item={
                "video_model_prompt": "cut to image 2",
                "segmind_payload_status": "ready",
                "segmind_payload": {
                    "prompt": "cut to image 2",
                    "duration": 4,
                    "generate_audio": True,
                    "reference_images": ["https://example.com/asset.jpg", "https://example.com/ref.jpg"],
                },
                "selected_assets": ["/tmp/character.png"],
                "referenced_frame_paths": ["/tmp/ref.jpg"],
                "referenced_frame_labels": ["@ref1"],
            },
            api_key="test-key",
            cache=None,
        )

        self.assertEqual(5, payload["duration"])
        self.assertFalse(payload["generate_audio"])
        self.assertIn("REFERENCE IMAGE MAP:", payload["prompt"])
        self.assertIn("image 1: character sheet from character.png", payload["prompt"])
        self.assertIn("image 2: referenced cutaway/reaction frame from ref.jpg", payload["prompt"])
        self.assertNotIn("@ref1", payload["prompt"])

    def test_prepared_payload_drops_reference_audio_when_audio_generation_off(self):
        payload = build_segmind_payload(
            item={
                "video_model_prompt": "prompt",
                "generate_audio": False,
                "segmind_payload_status": "ready",
                "segmind_payload": {
                    "prompt": "prompt",
                    "duration": 1,
                    "generate_audio": True,
                    "reference_audios": ["https://example.com/short-audio.mp3"],
                },
            },
            api_key="test-key",
            cache=None,
        )

        self.assertEqual(5, payload["duration"])
        self.assertFalse(payload["generate_audio"])
        self.assertNotIn("reference_audios", payload)

    def test_build_segmind_payload_does_not_upload_audio_by_default(self):
        cache = mock.Mock()
        payload = build_segmind_payload(
            item={
                "video_model_prompt": "prompt",
                "audio_url": "https://example.com/audio.mp3",
                "generate_audio": True,
                "allow_reference_audio": True,
            },
            api_key="test-key",
            cache=cache,
        )

        self.assertNotIn("reference_audios", payload)
        cache.get.assert_not_called()

    def test_build_segmind_payload_uploads_audio_when_explicitly_enabled(self):
        cache = mock.Mock()
        cache.get.return_value = None
        cache.upload_audio.return_value = "https://cdn.example.com/audio.mp3"

        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "trimmed.mp3"
            audio_path.write_bytes(b"audio bytes")
            with mock.patch.dict(os.environ, {"LOKA_ENABLE_REFERENCE_AUDIO_UPLOAD": "1"}):
                payload = build_segmind_payload(
                    item={
                        "video_model_prompt": "prompt",
                        "audio_reference_path": str(audio_path),
                        "generate_audio": True,
                        "allow_reference_audio": True,
                    },
                    api_key="test-key",
                    cache=cache,
                )

        self.assertEqual(["https://cdn.example.com/audio.mp3"], payload["reference_audios"])
        cache.upload_audio.assert_called_once()
        cache.upsert.assert_called_once()

    @mock.patch("scripts.generate_seedance_video.SegmindClient")
    def test_create_seedance_task_uses_segmind_sdk_and_downloads_output(self, mock_client_class):
        class FakeResponse:
            headers = {"content-type": "video/mp4"}

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def raise_for_status(self):
                return None

            def iter_content(self, chunk_size):
                yield b"video-"
                yield b"bytes"

        fake_job = mock.Mock()
        fake_job.request_id = "segmind-request-123"
        fake_job.wait.return_value = {
            "status": "COMPLETED",
            "output": "https://example.com/generated.mp4",
        }
        fake_client = mock.Mock()
        fake_client.submit_async.return_value = fake_job
        mock_client_class.return_value = fake_client
        fake_session = mock.Mock()
        fake_session.get.return_value = FakeResponse()

        result = create_seedance_task(
            api_key="test-key",
            payload={"prompt": "make video", "duration": 5},
            session=fake_session,
        )

        mock_client_class.assert_called_once_with(api_key="test-key", timeout=60.0)
        fake_client.submit_async.assert_called_once_with(
            "seedance-2.0",
            prompt="make video",
            duration=5,
        )
        fake_job.wait.assert_called_once_with(timeout=1800, interval=2.0)
        fake_session.get.assert_called_once_with(
            "https://example.com/generated.mp4",
            stream=True,
            timeout=300,
        )
        self.assertEqual("segmind-request-123", result["id"])
        self.assertEqual(b"video-bytes", result["content"]["bytes"])
        self.assertEqual("https://example.com/generated.mp4", result["content"]["video_url"])


if __name__ == "__main__":
    unittest.main()
