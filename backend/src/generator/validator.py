import re
import os
from typing import List, Dict, Any, Optional
from ..schemas import QualityCheckResult
from .client import Provider, generate_structured
from ..utils import get_prompt_template

FORBIDDEN_WORDS = [
    r"\bepic\b",
    r"\blots\s+of\s+movement\b",
]

NEGATIVE_PATTERNS = [
    r"\bno\s+distortion\b",
    r"\bno\s+deformation\b",
    r"\bno\s+warping\b",
    r"\bno\s+flicker\b",
]

def run_quality_check(
    provider: Provider,
    client,
    model: str,
    prompt: str,
    feedback_items: List[Dict[str, Any]],
    selected_assets: List[str],
    has_clip: bool = True,
) -> Dict[str, Any]:
    """Perform deterministic and semantic quality checks on the generated prompt."""
    
    # 1. Deterministic checks (Regex & word count)
    word_count = len(prompt.split())
    suggestions = []
    forbidden_found = []
    
    if word_count < 150:
        suggestions.append(f"Prompt is too short ({word_count} words). Expand with more visual details to reach at least 150 words.")
        
    prompt_lower = prompt.lower()
    
    # Check for forbidden words
    for pat in FORBIDDEN_WORDS:
        match = re.search(pat, prompt_lower)
        if match:
            word = match.group(0)
            forbidden_found.append(word)
            suggestions.append(f"Remove forbidden word '{word}' and replace with concrete motion description.")
            
    # Check for multiple/unqualified "fast"
    fast_count = len(re.findall(r"\bfast\b", prompt_lower))
    if fast_count > 1:
        suggestions.append("The term 'fast' is used multiple times. Seedance is sensitive to speed; make ONLY ONE element fast and ensure others are controlled.")
        forbidden_found.append("multiple 'fast'")
        
    # Check for negative patterns
    for pat in NEGATIVE_PATTERNS:
        match = re.search(pat, prompt_lower)
        if match:
            word = match.group(0)
            forbidden_found.append(word)
            suggestions.append(f"Remove negative constraint '{word}'. Rephrase as positive (e.g. 'face clear and undistorted, features correct').")
            
    # Check reference handles
    for idx in range(len(selected_assets)):
        handle = f"@image{idx + 1}"
        if handle not in prompt:
            suggestions.append(f"Reference asset handle '{handle}' was selected but is not explicitly mentioned in the prompt text.")
            
    if has_clip and "@video1" not in prompt:
        suggestions.append("Original clip handle '@video1' is available but not explicitly mentioned in the prompt text.")
        
    # 2. Semantic check (LLM-based)
    feedback_text = "\n".join([f"- {item.get('remark', '')}" for item in feedback_items])
    asset_text = "\n".join([f"- @image{idx+1}: {os.path.basename(path)}" for idx, path in enumerate(selected_assets)])
    
    eval_prompt = (
        f"Analyze this generated Seedance 2.0 prompt for compliance with feedback, wardrobe continuity, and style rules.\n\n"
        f"CLIENT FEEDBACK:\n{feedback_text or 'None'}\n\n"
        f"SELECTED ASSET REFERENCE LEGEND:\n{asset_text or 'None'}\n\n"
        f"GENERATED AI VIDEO PROMPT:\n{prompt}\n\n"
        f"Instructions:\n"
        f"1. Evaluate if every feedback remark is addressed (especially vague ones like 'looks disconnected' or 'make playful').\n"
        f"2. Check if clothing/wardrobe is explicitly described, prioritizing the clothing from the video clip frames (@video1) if there's a discrepancy between references.\n"
        f"3. Evaluate if vague feedback has been translated into concrete actions in the prompt.\n"
        f"4. Provide constructive suggestions to fix any issues."
    )
    
    try:
        response_json = generate_structured(
            provider=provider,
            client=client,
            model=model,
            contents=[eval_prompt],
            schema=QualityCheckResult,
            system_instruction=get_prompt_template(
                "quality_evaluator_system",
                "You are an expert film editor and prompt QA reviewer. You evaluate generated video prompts for absolute adherence to feedback, clothing consistency, and scene continuity."
            ),
            temperature=0.1
        )
        
        # Merge deterministic findings into LLM findings
        llm_forbidden = response_json.get("forbidden_terms_found", [])
        combined_forbidden = list(set(forbidden_found + (llm_forbidden or [])))
        
        llm_suggestions = response_json.get("suggestions", [])
        combined_suggestions = suggestions + [s for s in (llm_suggestions or []) if s not in suggestions]
        
        passed = response_json.get("passed", False)
        if len(suggestions) > 0:
            passed = False
            
        return {
            "passed": passed,
            "feedback_adherence": response_json.get("feedback_adherence", ""),
            "clothing_consistency": response_json.get("clothing_consistency", ""),
            "forbidden_terms_found": combined_forbidden,
            "vague_feedback_resolution": response_json.get("vague_feedback_resolution", ""),
            "suggestions": combined_suggestions,
        }
    except Exception as exc:
        print(f"Warning: Quality evaluator LLM check failed: {exc}")
        return {
            "passed": len(suggestions) == 0,
            "feedback_adherence": "Skipped (LLM evaluator failed)",
            "clothing_consistency": "Skipped (LLM evaluator failed)",
            "forbidden_terms_found": forbidden_found,
            "vague_feedback_resolution": "Skipped (LLM evaluator failed)",
            "suggestions": suggestions + [f"LLM evaluator error: {exc}"],
        }
