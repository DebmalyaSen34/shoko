import os
import json
import unittest
import shutil
import tempfile
from pathlib import Path
import server
from server import save_output_to_prompts

class TestHistoryAndErrors(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.project_name = "test_history_proj"
        self.project_dir = os.path.join(self.temp_dir, self.project_name)
        os.makedirs(self.project_dir, exist_ok=True)
        
        self.output_json = os.path.join(self.project_dir, "output.json")
        self.prompts_json = os.path.join(self.project_dir, "video_prompts.json")
        
        self.real_project_name = "test-history-temp-project"
        self.real_project_dir = f"data/{self.real_project_name}"
        os.makedirs(self.real_project_dir, exist_ok=True)
        self.real_output_json = f"{self.real_project_dir}/output.json"
        self.real_prompts_json = f"{self.real_project_dir}/video_prompts.json"
        
        self.original_data_dir = server.DATA_DIR
        server.DATA_DIR = Path("data")
        
    def tearDown(self):
        server.DATA_DIR = self.original_data_dir
        if os.path.exists(self.real_project_dir):
            shutil.rmtree(self.real_project_dir)

    def test_save_output_to_prompts_success_and_history(self):
        if os.path.exists(self.real_prompts_json):
            os.remove(self.real_prompts_json)
            
        success_output = [
            {
                "matched_clip": "clip1.mp4",
                "video_model_prompt": "First successful prompt plan for clip1",
                "selected_assets": ["char1.png"],
                "explanation": "Exp 1",
                "prompt_format": "complex",
                "applied_prompt_lessons": [
                    {
                        "id": "lesson-1",
                        "scope": "project",
                        "category": "continuity_error",
                        "lesson": "Preserve wardrobe.",
                    }
                ],
            }
        ]
        with open(self.real_output_json, "w", encoding="utf-8") as f:
            json.dump(success_output, f)
            
        save_output_to_prompts(self.real_project_name, "openai")
        
        self.assertTrue(os.path.exists(self.real_prompts_json))
        with open(self.real_prompts_json, "r", encoding="utf-8") as f:
            prompts = json.load(f)
            
        self.assertEqual(len(prompts), 1)
        item = prompts[0]
        self.assertEqual(item["clip_used"], "clip1.mp4")
        self.assertEqual(item["video_model_prompt"], "First successful prompt plan for clip1")
        self.assertEqual(item.get("latest_error"), None)
        self.assertEqual(len(item["history"]), 1)
        self.assertEqual(item["history"][0]["video_model_prompt"], "First successful prompt plan for clip1")
        self.assertEqual(item["history"][0]["provider"], "openai")
        self.assertEqual("lesson-1", item["applied_prompt_lessons"][0]["id"])
        self.assertEqual("lesson-1", item["history"][0]["applied_prompt_lessons"][0]["id"])

        second_output = [
            {
                "matched_clip": "clip1.mp4",
                "video_model_prompt": "Second successful prompt plan for clip1",
                "selected_assets": ["char1.png", "char2.png"],
                "explanation": "Exp 2",
                "prompt_format": "complex",
                "applied_prompt_lessons": [
                    {
                        "id": "lesson-2",
                        "scope": "clip",
                        "category": "too_vague",
                        "lesson": "Use concrete motion language.",
                    }
                ],
            }
        ]
        with open(self.real_output_json, "w", encoding="utf-8") as f:
            json.dump(second_output, f)
            
        save_output_to_prompts(self.real_project_name, "gemini")
        
        with open(self.real_prompts_json, "r", encoding="utf-8") as f:
            prompts = json.load(f)
            
        item = prompts[0]
        self.assertEqual(item["video_model_prompt"], "Second successful prompt plan for clip1")
        self.assertEqual(item.get("latest_error"), None)
        self.assertEqual(len(item["history"]), 2)
        self.assertEqual(item["history"][0]["video_model_prompt"], "First successful prompt plan for clip1")
        self.assertEqual(item["history"][1]["video_model_prompt"], "Second successful prompt plan for clip1")
        self.assertEqual(item["history"][1]["provider"], "gemini")
        self.assertEqual("lesson-2", item["applied_prompt_lessons"][0]["id"])
        self.assertEqual("lesson-2", item["history"][1]["applied_prompt_lessons"][0]["id"])

    def test_save_output_to_prompts_error_handling(self):
        initial_prompts = [
            {
                "clip_used": "clip1.mp4",
                "video_model_prompt": "Last successful prompt",
                "selected_assets": ["char1.png"],
                "explanation": "Exp initial",
                "status": "success",
                "history": [
                    {
                        "timestamp": "2026-07-13T10:00:00Z",
                        "provider": "openai",
                        "video_model_prompt": "Last successful prompt",
                        "selected_assets": ["char1.png"],
                        "explanation": "Exp initial"
                    }
                ]
            }
        ]
        with open(self.real_prompts_json, "w", encoding="utf-8") as f:
            json.dump(initial_prompts, f)
            
        error_output = [
            {
                "matched_clip": "clip1.mp4",
                "video_model_prompt": "API Error: Quota expired for this model usage",
                "selected_assets": [],
                "explanation": "",
                "prompt_format": "complex"
            }
        ]
        with open(self.real_output_json, "w", encoding="utf-8") as f:
            json.dump(error_output, f)
            
        save_output_to_prompts(self.real_project_name, "gemini")
        
        with open(self.real_prompts_json, "r", encoding="utf-8") as f:
            prompts = json.load(f)
            
        item = prompts[0]
        self.assertEqual(item["video_model_prompt"], "Last successful prompt")
        self.assertEqual(len(item["history"]), 1)
        self.assertEqual(item.get("latest_error"), "API Error: Quota expired for this model usage")

        success_output = [
            {
                "matched_clip": "clip1.mp4",
                "video_model_prompt": "Fresh successful prompt",
                "selected_assets": ["char3.png"],
                "explanation": "Exp fresh",
                "prompt_format": "complex"
            }
        ]
        with open(self.real_output_json, "w", encoding="utf-8") as f:
            json.dump(success_output, f)
            
        save_output_to_prompts(self.real_project_name, "openai")
        
        with open(self.real_prompts_json, "r", encoding="utf-8") as f:
            prompts = json.load(f)
            
        item = prompts[0]
        self.assertEqual(item["video_model_prompt"], "Fresh successful prompt")
        self.assertEqual(item.get("latest_error"), None)
        self.assertEqual(len(item["history"]), 2)
        self.assertEqual(item["history"][1]["video_model_prompt"], "Fresh successful prompt")
