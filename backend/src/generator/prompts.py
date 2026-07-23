import os
from typing import List, Dict, Any, Optional

from .media import _aspect_ratio
from .upload import _append_media_reference
from src.utils import app_resource_path

DEFAULT_PROJECT_OVERRIDES = """

---

# PROJECT OVERRIDES — HIGHEST PRIORITY

These rules override every conflicting instruction in the skill above:

- English output for all prompt instructions and descriptions, except preserve supplied Hindi/English dialogue text exactly when `DIALOGUE_FOR_LIP_SYNC` is present. Never produce Chinese, ZH, unrelated bilingual output, or translations.
- Never use timeline prompting, timestamp brackets, time-labelled beats, shot maps, or timed tables.
- Process visual feedback only. Do not invent or modify dialogue, voice, music, sound effects, or audio direction. When exact Hindi/English dialogue is supplied for lip sync, include it unchanged only as dialogue to be spoken/lip-synced.
- Generate one detailed, scene-specific Seedance 2.0 prompt per cluster, at least 150 English words.
- Choose `plain_text`, `json`, or `production_prompt` using the skill's scene routing, but remove timeline structures from every format.
- Preserve every client feedback instruction and all observable continuity from the original clip.
- Use the initial frame prompt as the first-frame visual anchor for the final video prompt.
- Use Segmind API reference wording consistently. In the final prompt body, refer to uploaded references as `image 1`, `image 2`, ... exactly like the Segmind documentation examples. Do not use `@image1`, `@video1`, bracketed `[Image 1]`, or custom names in the final prompt body.
- Use the supplied reference legend only as a mapping aid. Convert selected sheets from `@image1` through `@image9` into `image 1` through `image 9` when writing the final prompt. Convert original clip frames into the image numbers supplied by the runtime payload, or describe them as "the original clip reference images" when exact numbers are unavailable.
- Perform strict visual attribute bridging: when referencing image sheets (like `image 1` for characters or `image 2` for locations), do not rely solely on the image number. Explicitly extract and describe the key visual elements visible in the reference sheets (e.g., specific hair color, clothing style, accessories, colors, textures, lighting, setting details) directly in the generated prompt text to bind the generation model and prevent hallucinations.
- Reconcile clothing discrepancy: If the character's clothing in the preproduction assets differs from the clothing in the original video clip reference images, prioritize the clothing/wardrobe from the video clip to ensure continuity, unless the client feedback explicitly requests changing the wardrobe to match the preproduction asset. Explicitly describe this wardrobe in detail in the prompt body (e.g., color, garment type, fabric, fit) and make sure it matches the initial frame prompt exactly.
- Reconcile version variants: If multiple version sheets of a character are selected, compare all of them with the original video clip frames. Explicitly determine which version closely matches the character's clothing and appearance in the video clip, and use only that correct image number for Vir's identity/clothing reference in the prompt.
- Ensure body continuity: Analyze the character's physical proportions in the original video clip reference images, specifically their height, body shape, build, and posture. Explicitly describe these physical attributes in the prompt (e.g., 'a slender boy of average height', 'a broad-shouldered athletic build') and keep this description consistent between the initial frame prompt and the final video prompt to prevent body shape or proportions drift.
- Translate vague feedback: If a feedback remark is vague or abstract (e.g., 'make him playful', 'create proper freeze point', 'looks disconnected'), cross-reference it with the visual state in the original clip reference images and translate it into concrete, actionable staging instructions (e.g., specifying character poses, expressions, body physics, or camera movements) to resolve the vagueness, and document how this was clarified in the explanation.
- Select no more than nine image assets and include an English `reference_legend` matching labels used in the prompt.
- Return exactly one structured result for every supplied cluster_id.
"""


def _default_seedance_skill_text() -> str:
    skill_path = app_resource_path("skill", "video_generation_skill.md")
    with open(skill_path, "r", encoding="utf-8") as file:
        return file.read()


def _failed_batch_result(cluster_id: int, explanation: str) -> Dict[str, Any]:
    return {
        "cluster_id": cluster_id,
        "initial_frame_prompt": "",
        "initial_frame_image_path": "",
        "clip_frame_paths": [],
        "selected_assets": [],
        "prompt_format": None,
        "reference_legend": "",
        "video_model_prompt": f"Failed to generate: {explanation}",
        "explanation": explanation,
        "status": "failed",
        "quality_warning": None,
        "quality_report": None,
    }


def _cluster_prompt_context(cluster_id: int, cluster: Dict[str, Any]) -> str:
    clip = cluster.get("matched_clip")
    clip_name = clip.get("clip") if clip else None
    duration_s = clip.get("duration_s", 0.0) if clip else 0.0
    frame_size = cluster.get("frame_size", "unknown")
    aspect_ratio = _aspect_ratio(frame_size)
    feedback = "\n".join(
        f"- {item.get('remark', '')}"
        for item in cluster.get("feedback_items", [])
    )
    clip_context = cluster.get("clip_context") or {}
    context_text = ""
    if clip_context:
        context_fields = {
            key: clip_context.get(key)
            for key in [
                "summary",
                "visible_characters",
                "expressions",
                "gaze",
                "actions",
                "blocking",
                "camera_framing",
                "location",
                "continuity_notes",
                "uncertainty_flags",
                "status",
            ]
        }
        context_text = f"CLIP_UNDERSTANDING_CONTEXT:\n{context_fields}\n"
    dialogue_context = cluster.get("dialogue_context") or {}
    dialogue_text = (dialogue_context.get("dialogue_text") or "").strip()
    dialogue_prompt_text = ""
    if dialogue_context.get("has_reference_audio"):
        if dialogue_text:
            dialogue_prompt_text = (
                "DIALOGUE_FOR_LIP_SYNC:\n"
                f"- Language: {dialogue_context.get('dialogue_language') or 'unknown'}\n"
                f"- Spoken dialogue only: \"{dialogue_text}\"\n"
                "- Use this exact Hindi/English dialogue for mouth shapes, pauses, and delivery. "
                "Lip-sync to the supplied reference audio and this dialogue text. Do not add narration, "
                "translate other languages, or invent extra spoken lines.\n"
            )
        else:
            dialogue_prompt_text = (
                "DIALOGUE_FOR_LIP_SYNC:\n"
                "- No Hindi/English spoken character dialogue was extracted from the matching audio segment. "
                "Do not invent dialogue or use narration as spoken lines.\n"
            )
    return (
        f"CLUSTER_ID: {cluster_id}\n"
        f"CLIP: {clip_name or 'unknown'}\n"
        f"DURATION_SECONDS: {duration_s}\n"
        f"FRAME_SIZE: {frame_size}\n"
        f"ASPECT_RATIO: {aspect_ratio}\n"
        f"{context_text}"
        f"{dialogue_prompt_text}"
        f"VIDEO_FEEDBACK:\n{feedback}\n"
        "The original video for this cluster follows when available."
    )


def _reference_asset_manifest(reference_assets: List[str]) -> str:
    if not reference_assets:
        return "No reference assets are available."
    return "\n".join(f"- {path}" for path in reference_assets)


def _resolve_selected_asset_paths(
    selected_assets: List[str],
    reference_assets: List[str],
) -> List[str]:
    path_lookup = {}
    for path in reference_assets:
        path_lookup[path] = path
        path_lookup[os.path.abspath(path)] = path
        path_lookup[os.path.basename(path)] = path

    resolved = []
    seen = set()
    for selected in selected_assets:
        candidate = str(selected or "").strip()
        if not candidate:
            continue
        path = path_lookup.get(candidate) or path_lookup.get(os.path.abspath(candidate))
        if path and path not in seen:
            resolved.append(path)
            seen.add(path)

    # Expand versioned assets if any character/asset has multiple versions (v1, v2)
    import re
    version_pattern = re.compile(r"^(.*?)-v\d+")
    
    expanded = list(resolved)
    for path in resolved:
        filename = os.path.basename(path)
        match = version_pattern.match(filename)
        if match:
            prefix = match.group(1)
            for cand in reference_assets:
                cand_filename = os.path.basename(cand)
                cand_match = version_pattern.match(cand_filename)
                if cand_match and cand_match.group(1) == prefix:
                    if cand not in seen:
                        expanded.append(cand)
                        seen.add(cand)

    return expanded[:9]


def _reference_legend_for_assets(selected_assets: List[str]) -> str:
    legend_lines = []
    for index, path in enumerate(selected_assets, start=1):
        basename = os.path.basename(path)
        legend_lines.append(f"image {index} — {basename} — selected visual reference.")
    legend_lines.append("original clip reference images — motion, framing, and continuity reference.")
    return "\n".join(legend_lines)


def _append_selected_reference_assets(
    contents: List[Any],
    selected_assets: List[str],
    reference_handles: Dict[str, Any],
) -> None:
    if not selected_assets:
        contents.append(
            "SELECTED_REFERENCE_ASSETS: None. Use the original clip frames as the "
            "primary visual continuity reference."
        )
        return

    contents.append(
        "SELECTED_REFERENCE_ASSETS: Only the following labeled image references "
        "are available for this cluster. Use these exact labels if references are "
        "mentioned in the prompt or legend."
    )
    for index, path in enumerate(selected_assets, start=1):
        contents.append(f"image {index}: REFERENCE_ASSET_PATH: {path}")
        ref = reference_handles.get(path)
        if ref:
            _append_media_reference(contents, ref)


def _append_clip_reference(
    contents: List[Any],
    assets_dir: str,
    clip_handles: Dict[str, Any],
    cluster: Dict[str, Any],
    cluster_id: Optional[int] = None,
) -> None:
    clip = cluster.get("matched_clip")
    clip_name = clip.get("clip") if clip else None
    if not clip_name:
        return
    path = os.path.join(assets_dir, "06_clips", "_raw", clip_name)
    if path in clip_handles:
        ref = clip_handles[path]
    elif cluster_id is not None and cluster_id in clip_handles:
        ref = clip_handles[cluster_id]
    else:
        return

    if isinstance(ref, list):
        contents.append(
            "ORIGINAL_CLIP_FRAMES: The following still frames were extracted "
            "from the original clip in chronological order across the full clip."
        )
        for frame_ref in ref:
            _append_media_reference(contents, frame_ref)
    else:
        _append_media_reference(contents, ref)
