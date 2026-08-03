import unittest
from unittest import mock
from src.generator.validator import merge_learning_eval_report, run_learning_eval, run_quality_check

class ValidatorTests(unittest.TestCase):
    def test_word_count_check(self):
        # Prompt under 150 words
        short_prompt = "A simple prompt with few words."
        res = run_quality_check(
            provider="gemini",
            client=mock.MagicMock(),
            model="mock-model",
            prompt=short_prompt,
            feedback_items=[],
            selected_assets=[],
            has_clip=False
        )
        self.assertFalse(res["passed"])
        self.assertTrue(any("too short" in s for s in res["suggestions"]))

    def test_forbidden_words_check(self):
        # Prompt containing forbidden and negative words
        bad_prompt = (
            "This is a very long prompt to bypass the word count limit. " * 20 +
            "The camera moves epic and fast and fast again with lots of movement, showing no distortion and no flicker. "
        )
        res = run_quality_check(
            provider="gemini",
            client=mock.MagicMock(),
            model="mock-model",
            prompt=bad_prompt,
            feedback_items=[],
            selected_assets=[],
            has_clip=False
        )
        self.assertFalse(res["passed"])
        self.assertIn("epic", res["forbidden_terms_found"])
        self.assertIn("lots of movement", res["forbidden_terms_found"])
        self.assertIn("multiple 'fast'", res["forbidden_terms_found"])
        self.assertIn("no distortion", res["forbidden_terms_found"])
        self.assertIn("no flicker", res["forbidden_terms_found"])

    def test_asset_handles_check(self):
        # Prompt missing required asset handles
        prompt = "This is a very long prompt to bypass the word count limit. " * 25
        res = run_quality_check(
            provider="gemini",
            client=mock.MagicMock(),
            model="mock-model",
            prompt=prompt,
            feedback_items=[],
            selected_assets=["char.png", "loc.png"],
            has_clip=True
        )
        self.assertFalse(res["passed"])
        self.assertTrue(any("image 1" in s for s in res["suggestions"]))
        self.assertTrue(any("image 2" in s for s in res["suggestions"]))
        self.assertTrue(any("Original clip reference images" in s for s in res["suggestions"]))

    @mock.patch("src.generator.validator.generate_structured")
    def test_semantic_checks(self, mock_generate_structured):
        # Mock semantic LLM evaluation response
        mock_generate_structured.return_value = {
            "passed": True,
            "feedback_adherence": "All feedback items addressed perfectly.",
            "clothing_consistency": "Clothing matches video frames.",
            "forbidden_terms_found": [],
            "vague_feedback_resolution": "Vague feedback resolved by panning camera slowly.",
            "suggestions": []
        }
        
        valid_prompt = (
            "A long prompt with image 1 and image 2 that starts with original clip reference images and describes a character wearing a green shirt. " * 15
        )
        
        res = run_quality_check(
            provider="gemini",
            client=mock.MagicMock(),
            model="mock-model",
            prompt=valid_prompt,
            feedback_items=[{"remark": "make it playful"}],
            selected_assets=["char.png", "loc.png"],
            has_clip=True
        )
        
        self.assertTrue(res["passed"])
        self.assertEqual("All feedback items addressed perfectly.", res["feedback_adherence"])
        self.assertEqual("Clothing matches video frames.", res["clothing_consistency"])
        self.assertEqual("Vague feedback resolved by panning camera slowly.", res["vague_feedback_resolution"])
        mock_generate_structured.assert_called_once()

    @mock.patch("src.generator.orchestrator.generate_structured")
    def test_prompt_refinement(self, mock_generate_structured):
        from src.generator.orchestrator import _refine_prompt
        
        mock_generate_structured.return_value = {
            "english_prompt": "A refined prompt that complies with suggestions.",
            "refinement_explanation": "Rephrased negative constraints."
        }
        
        res = _refine_prompt(
            provider="gemini",
            client=mock.MagicMock(),
            model="mock-model",
            draft_prompt="A draft prompt with issues.",
            feedback_items=[{"remark": "make it playful"}],
            suggestions=["Remove negative terms"]
        )
        
        self.assertEqual("A refined prompt that complies with suggestions.", res)
        mock_generate_structured.assert_called_once()

    def test_resolve_selected_asset_paths_expansion(self):
        from src.generator.prompts import _resolve_selected_asset_paths
        
        reference_assets = [
            "/path/to/vir-sheet-v1.png",
            "/path/to/vir-sheet-v2.png",
            "/path/to/vir-sheet-v3.png",
            "/path/to/unused-rian.png",
            "/path/to/location-v1.png",
            "/path/to/location-v2.png",
        ]
        
        # When only v1 is selected, it should expand to include v2 and v3
        selected = ["vir-sheet-v1.png", "location-v1.png"]
        resolved = _resolve_selected_asset_paths(selected, reference_assets)
        
        self.assertIn("/path/to/vir-sheet-v1.png", resolved)
        self.assertIn("/path/to/vir-sheet-v2.png", resolved)
        self.assertIn("/path/to/vir-sheet-v3.png", resolved)
        self.assertIn("/path/to/location-v1.png", resolved)
        self.assertIn("/path/to/location-v2.png", resolved)
        self.assertNotIn("/path/to/unused-rian.png", resolved)

    def test_learning_eval_deterministic_checks(self):
        res = run_learning_eval(
            provider="gemini",
            client=mock.MagicMock(),
            model="mock-model",
            prompt="Short prompt with @image1 and no wardrobe.",
            eval_cases=[
                {
                    "id": "case-1",
                    "name": "Wardrobe continuity",
                    "input": {
                        "feedback_items": [{"remark": "Preserve slow hand movement and wardrobe."}],
                        "clip_summary": "Boy in blue shirt raises hand.",
                        "selected_assets": ["char.png"],
                    },
                    "expected_behavior": [
                        "Must preserve visible wardrobe.",
                        "Must mention slow deliberate hand movement.",
                    ],
                    "failure_categories": ["continuity_error"],
                    "enabled": True,
                }
            ],
            lessons=[],
        )

        self.assertFalse(res["passed"])
        self.assertIn("case-1", res["failed_cases"])
        self.assertTrue(any("too short" in suggestion for suggestion in res["suggestions"]))
        self.assertTrue(any("provider-incompatible" in suggestion for suggestion in res["suggestions"]))

    @mock.patch("src.generator.validator.generate_structured")
    def test_learning_eval_merges_llm_judge_result(self, mock_generate_structured):
        mock_generate_structured.return_value = {
            "passed": False,
            "score": 0.45,
            "failed_cases": ["case-llm"],
            "suggestions": ["Prompt does not explicitly preserve wardrobe continuity."],
            "case_results": [{"case_id": "case-llm", "passed": False, "score": 0.45}],
        }
        prompt = (
            "A detailed prompt with image 1 and original clip reference images that must preserve visible wardrobe and slow deliberate hand movement. "
            * 15
        )

        res = run_learning_eval(
            provider="gemini",
            client=mock.MagicMock(),
            model="mock-model",
            prompt=prompt,
            eval_cases=[
                {
                    "id": "case-llm",
                    "name": "Wardrobe continuity",
                    "input": {"feedback_items": [], "clip_summary": "", "selected_assets": ["char.png"]},
                    "expected_behavior": ["Must preserve visible wardrobe."],
                    "failure_categories": ["continuity_error"],
                    "enabled": True,
                }
            ],
            lessons=[],
        )

        self.assertFalse(res["passed"])
        self.assertEqual(["case-llm"], res["failed_cases"])
        self.assertIn("Prompt does not explicitly preserve wardrobe continuity.", res["suggestions"])

    @mock.patch("src.generator.validator.generate_structured", side_effect=RuntimeError("judge down"))
    def test_learning_eval_judge_failure_is_non_blocking(self, _mock_generate_structured):
        prompt = (
            "A detailed prompt with image 1 and original clip reference images that must preserve visible wardrobe and slow deliberate hand movement. "
            * 15
        )

        res = run_learning_eval(
            provider="gemini",
            client=mock.MagicMock(),
            model="mock-model",
            prompt=prompt,
            eval_cases=[
                {
                    "id": "case-ok",
                    "name": "Wardrobe continuity",
                    "input": {"feedback_items": [], "clip_summary": "", "selected_assets": ["char.png"]},
                    "expected_behavior": ["Must preserve visible wardrobe."],
                    "failure_categories": ["continuity_error"],
                    "enabled": True,
                }
            ],
            lessons=[],
        )

        self.assertTrue(res["passed"])
        self.assertTrue(any("judge error" in suggestion.lower() for suggestion in res["suggestions"]))

    def test_failed_learning_eval_marks_quality_report_failed(self):
        merged = merge_learning_eval_report(
            {"passed": True, "suggestions": []},
            {
                "passed": False,
                "score": 0.5,
                "failed_cases": ["case-1"],
                "case_results": [],
                "suggestions": ["Preserve wardrobe continuity."],
            },
        )

        self.assertFalse(merged["passed"])
        self.assertEqual(["case-1"], merged["learning_eval"]["failed_cases"])
        self.assertIn("Preserve wardrobe continuity.", merged["suggestions"])
