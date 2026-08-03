import re
import os
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from ..schemas import QualityCheckResult
from .client import Provider, generate_structured
from ..utils import get_prompt_template
from src.prompt_learning import tokenize

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

SEGMIND_HANDLE_PATTERN = r"@(?:image|video|ref)\d+"


class LearningEvalJudgeResult(BaseModel):
    passed: bool = Field(description="True if the prompt satisfies the supplied learning eval cases.")
    score: float = Field(description="Score from 0 to 1 for learning eval compliance.")
    failed_cases: List[str] = Field(description="IDs of eval cases that failed.")
    suggestions: List[str] = Field(description="Actionable suggestions for fixing learning eval failures.")
    case_results: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Per-case judgement details with case id, passed flag, score, and reasoning.",
    )


def _mentions_image_reference(prompt_lower: str, index: int) -> bool:
    return re.search(rf"\bimage\s+{index}\b", prompt_lower) is not None


def _mentions_original_clip_reference(prompt_lower: str) -> bool:
    return (
        "original clip" in prompt_lower
        or "original clip reference images" in prompt_lower
    )


def _deterministic_learning_case_check(prompt: str, case: Dict[str, Any]) -> Dict[str, Any]:
    prompt_lower = prompt.lower()
    suggestions = []
    failed_checks = []

    word_count = len(prompt.split())
    if word_count < 150:
        failed_checks.append("prompt_length")
        suggestions.append(f"Eval case {case.get('id')}: prompt is too short ({word_count} words).")

    forbidden_found = []
    for pat in FORBIDDEN_WORDS + NEGATIVE_PATTERNS:
        for match in re.finditer(pat, prompt_lower):
            forbidden_found.append(match.group(0))
    stale_handles = sorted(set(re.findall(SEGMIND_HANDLE_PATTERN, prompt_lower)))
    forbidden_found.extend(stale_handles)
    if forbidden_found:
        failed_checks.append("forbidden_words")
        suggestions.append(
            f"Eval case {case.get('id')}: remove forbidden or provider-incompatible terms: {', '.join(sorted(set(forbidden_found)))}."
        )

    selected_assets = ((case.get("input") or {}).get("selected_assets") or [])
    missing_assets = []
    for index, _asset in enumerate(selected_assets, start=1):
        if not _mentions_image_reference(prompt_lower, index):
            missing_assets.append(f"image {index}")
    if missing_assets:
        failed_checks.append("missing_selected_asset_references")
        suggestions.append(
            f"Eval case {case.get('id')}: mention selected asset references {', '.join(missing_assets)} explicitly."
        )

    expected_terms = tokenize(" ".join(case.get("expected_behavior") or []))
    present_expected_terms = {term for term in expected_terms if term in prompt_lower}
    if expected_terms and len(present_expected_terms) < max(1, min(3, len(expected_terms))):
        failed_checks.append("missing_required_terms")
        missing = sorted(expected_terms - present_expected_terms)[:6]
        suggestions.append(
            f"Eval case {case.get('id')}: include expected behavior terms such as {', '.join(missing)}."
        )

    feedback_text = " ".join(
        str(item.get("remark") or item.get("comment") or item.get("correction") or "")
        for item in ((case.get("input") or {}).get("feedback_items") or [])
        if isinstance(item, dict)
    )
    feedback_terms = tokenize(feedback_text)
    present_feedback_terms = {term for term in feedback_terms if term in prompt_lower}
    if feedback_terms and len(present_feedback_terms) < max(1, min(2, len(feedback_terms))):
        failed_checks.append("missing_feedback_keywords")
        missing = sorted(feedback_terms - present_feedback_terms)[:6]
        suggestions.append(
            f"Eval case {case.get('id')}: reflect feedback keywords such as {', '.join(missing)}."
        )

    passed = not failed_checks
    return {
        "case_id": case.get("id"),
        "passed": passed,
        "score": 1.0 if passed else max(0.0, 1.0 - (len(failed_checks) * 0.2)),
        "failed_checks": failed_checks,
        "suggestions": suggestions,
    }


def run_learning_eval(
    provider: Provider,
    client,
    model: str,
    prompt: str,
    eval_cases: List[Dict[str, Any]],
    lessons: List[Dict[str, Any]],
) -> Dict[str, Any]:
    enabled_cases = [case for case in eval_cases or [] if case.get("enabled", True)]
    if not enabled_cases:
        return {
            "passed": True,
            "score": 1.0,
            "failed_cases": [],
            "case_results": [],
            "suggestions": [],
        }

    deterministic_results = [
        _deterministic_learning_case_check(prompt, case)
        for case in enabled_cases
    ]
    deterministic_suggestions = [
        suggestion
        for result in deterministic_results
        for suggestion in result.get("suggestions", [])
    ]
    failed_cases = {
        str(result.get("case_id"))
        for result in deterministic_results
        if not result.get("passed") and result.get("case_id")
    }
    deterministic_score = (
        sum(float(result.get("score", 0.0)) for result in deterministic_results) / len(deterministic_results)
    )

    lesson_text = "\n".join(f"- {lesson.get('lesson')}" for lesson in lessons or [] if lesson.get("lesson"))
    judge_prompt = (
        "Evaluate this generated video prompt against saved regression eval cases. "
        "Judge only the requested behavior; do not reward unrelated detail.\n\n"
        f"SAVED LESSONS:\n{lesson_text or 'None'}\n\n"
        f"EVAL CASES:\n{enabled_cases}\n\n"
        f"GENERATED PROMPT:\n{prompt}\n\n"
        "Check whether the prompt addresses all expected behavior, avoids unsupported visual details, preserves continuity, "
        "and remains provider-compatible. Return per-case results and actionable suggestions."
    )

    judge_result = None
    judge_error = None
    try:
        judge_result = generate_structured(
            provider=provider,
            client=client,
            model=model,
            contents=[judge_prompt],
            schema=LearningEvalJudgeResult,
            system_instruction=get_prompt_template(
                "learning_eval_judge_system",
                "You are a strict regression evaluator for AI video prompts. You judge generated prompts against saved eval cases and return concise actionable failures.",
            ),
            temperature=0.1,
        )
    except Exception as exc:
        judge_error = str(exc)

    suggestions = list(deterministic_suggestions)
    case_results = list(deterministic_results)
    score = deterministic_score
    if judge_result:
        score = min(score, max(0.0, min(1.0, float(judge_result.get("score", score)))))
        failed_cases.update(str(case_id) for case_id in judge_result.get("failed_cases", []) if case_id)
        suggestions.extend(
            suggestion
            for suggestion in judge_result.get("suggestions", [])
            if suggestion not in suggestions
        )
        case_results.extend(judge_result.get("case_results", []) or [])
    elif judge_error:
        suggestions.append(f"Learning eval judge error: {judge_error}")

    passed = not failed_cases and score >= 0.8
    return {
        "passed": passed,
        "score": round(score, 4),
        "failed_cases": sorted(failed_cases),
        "case_results": case_results,
        "suggestions": suggestions,
    }


def merge_learning_eval_report(quality_report: Dict[str, Any], learning_eval: Dict[str, Any]) -> Dict[str, Any]:
    if not learning_eval or learning_eval.get("passed", True):
        if learning_eval:
            quality_report["learning_eval"] = learning_eval
        return quality_report
    quality_report = dict(quality_report)
    quality_report["passed"] = False
    quality_report["learning_eval"] = learning_eval
    existing = quality_report.get("suggestions") or []
    quality_report["suggestions"] = existing + [
        suggestion
        for suggestion in learning_eval.get("suggestions", [])
        if suggestion not in existing
    ]
    return quality_report

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
            
    # Check for stale provider handles that are not valid Segmind prompt wording.
    stale_handles = sorted(set(re.findall(SEGMIND_HANDLE_PATTERN, prompt_lower)))
    if stale_handles:
        forbidden_found.extend(stale_handles)
        suggestions.append(
            "Replace all @image/@video/@ref handles with Segmind wording such as image 1, image 2, or original clip reference images."
        )

    # Check reference mentions
    for idx in range(len(selected_assets)):
        image_number = idx + 1
        if not _mentions_image_reference(prompt_lower, image_number):
            suggestions.append(
                f"Reference asset 'image {image_number}' was selected but is not explicitly mentioned in the prompt text."
            )
            
    if has_clip and not _mentions_original_clip_reference(prompt_lower):
        suggestions.append("Original clip reference images are available but not explicitly mentioned in the prompt text.")
        
    # 2. Semantic check (LLM-based)
    feedback_text = "\n".join([f"- {item.get('remark', '')}" for item in feedback_items])
    asset_text = "\n".join([f"- image {idx+1}: {os.path.basename(path)}" for idx, path in enumerate(selected_assets)])
    
    eval_prompt = (
        f"Analyze this generated Seedance 2.0 prompt for compliance with feedback, wardrobe continuity, and style rules.\n\n"
        f"CLIENT FEEDBACK:\n{feedback_text or 'None'}\n\n"
        f"SELECTED ASSET REFERENCE LEGEND:\n{asset_text or 'None'}\n\n"
        f"GENERATED AI VIDEO PROMPT:\n{prompt}\n\n"
        f"Instructions:\n"
        f"1. Evaluate if every feedback remark is addressed (especially vague ones like 'looks disconnected' or 'make playful').\n"
        f"2. Check if clothing/wardrobe is explicitly described, prioritizing the clothing from the original clip reference images if there's a discrepancy between references.\n"
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
