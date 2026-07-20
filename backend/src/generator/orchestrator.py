import os
import time
import json
import base64
import re
from typing import List, Dict, Any, Optional
from google import genai
from openai import OpenAI
from google.genai import types

from config.settings import (
    LITE_MODEL,
    OPENAI_IMAGE_MODEL,
    REASONING_MODEL,
)
from ..schemas import (
    PromptResult,
    AudioInstructionResult,
    BatchInitialFramePromptResult,
    BatchPromptResult,
    BatchSelectedAssetsResult,
    RefinedPromptResult,
)
from ..utils import get_prompt_template

from . import media
from . import upload
from .media import _aspect_ratio, _image_size_for_aspect_ratio
from .client import Provider, _detect_provider, _default_model_for_provider, generate_structured
from .upload import (
    OpenAIFileReference,
    _openai_file_reference,
    _append_media_reference,
)
from .prompts import (
    DEFAULT_PROJECT_OVERRIDES,
    _default_seedance_skill_text,
    _failed_batch_result,
    _cluster_prompt_context,
    _reference_asset_manifest,
    _resolve_selected_asset_paths,
    _reference_legend_for_assets,
    _append_selected_reference_assets,
    _append_clip_reference,
)
from .validator import run_quality_check


def _openai_input_from_contents(contents: List[Any]) -> List[Dict[str, Any]]:
    # Re-use or adapt for batch usage if needed, or import from .client
    # Let's import it from .client to keep client-related input parsing centralized
    from .client import _openai_input_from_contents
    return _openai_input_from_contents(contents)


def _select_reference_assets_for_batch(
    *,
    provider: Provider,
    client,
    model: str,
    batch: List[tuple[int, Dict[str, Any]]],
    reference_assets: List[str],
) -> Dict[int, List[str]]:
    if not reference_assets:
        return {cluster_id: [] for cluster_id, _ in batch}

    contents = [
        (
            "Select the minimal character/location reference images needed for "
            "each clip cluster before any images are uploaded. Use only paths "
            "from REFERENCE_ASSET_CANDIDATES. Prefer the original clip frames "
            "for pose, framing, action, and room continuity; select asset images "
            "only for identity, wardrobe, props, or named locations that are "
            "explicitly needed. Return at most 3 assets per cluster, and never "
            "select raw clips or audio."
        ),
        f"REFERENCE_ASSET_CANDIDATES:\n{_reference_asset_manifest(reference_assets)}",
    ]
    batch_ids = set()
    for cluster_id, cluster in batch:
        batch_ids.add(cluster_id)
        contents.append(_cluster_prompt_context(cluster_id, cluster))

    response_json = generate_structured(
        provider=provider,
        client=client,
        model=model,
        contents=contents,
        schema=BatchSelectedAssetsResult,
        system_instruction=get_prompt_template(
            "preflight_asset_selector_system",
            "You are a precise preflight asset selector. Return only file paths "
            "from the supplied candidate list that are needed for each cluster."
        ),
        temperature=0.1,
    )

    selected_by_id = {cluster_id: [] for cluster_id, _ in batch}
    for item in response_json.get("results", []):
        cluster_id = item.get("cluster_id")
        if cluster_id not in batch_ids:
            continue
        selected_by_id[cluster_id] = _resolve_selected_asset_paths(
            item.get("selected_assets", []),
            reference_assets,
        )
    return selected_by_id


def _generate_openai_initial_frame_image(
    *,
    client: OpenAI,
    prompt: str,
    cluster_id: int,
    cluster: Dict[str, Any],
    initial_frames_dir: str,
) -> str:
    os.makedirs(initial_frames_dir, exist_ok=True)
    aspect_ratio = _aspect_ratio(cluster.get("frame_size", "unknown"))
    response = client.images.generate(
        model=os.environ.get("OPENAI_IMAGE_MODEL", OPENAI_IMAGE_MODEL),
        prompt=prompt,
        size=_image_size_for_aspect_ratio(aspect_ratio),
        quality=os.environ.get("OPENAI_IMAGE_QUALITY", "medium"),
        output_format="png",
    )
    if not response.data or not response.data[0].b64_json:
        raise RuntimeError("OpenAI image generation returned no image data")

    image_bytes = base64.b64decode(response.data[0].b64_json)
    image_path = os.path.join(initial_frames_dir, f"cluster_{cluster_id}_initial_frame.png")
    with open(image_path, "wb") as file:
        file.write(image_bytes)
    return image_path


def _cluster_media_dir_name(cluster_id: int, cluster: Dict[str, Any]) -> str:
    clip = cluster.get("matched_clip") or {}
    clip_name = os.path.splitext(os.path.basename(str(clip.get("clip") or "")))[0]
    occurrence = cluster.get("clip_occurrence")
    if occurrence is None:
        occurrence = cluster_id
    safe_clip = re.sub(r"[^A-Za-z0-9._-]+", "_", clip_name).strip("._-")
    return f"clip_{occurrence}_{safe_clip or f'cluster_{cluster_id}'}"


def _refine_prompt(
    *,
    provider: Provider,
    client,
    model: str,
    draft_prompt: str,
    feedback_items: List[Dict[str, Any]],
    suggestions: List[str],
) -> str:
    """Uses a text-only call to refine the draft prompt based on quality checker suggestions."""
    print("Draft prompt failed validation. Triggering text-only auto-refinement...")
    feedback_text = "\n".join([f"- {item.get('remark', '')}" for item in feedback_items])
    suggestions_text = "\n".join([f"- {sug}" for sug in suggestions])
    
    refiner_prompt = (
        f"You are refining a drafted AI video generation prompt to correct quality issues.\n\n"
        f"CLIENT FEEDBACK:\n{feedback_text or 'None'}\n\n"
        f"QUALITY CHECKER SUGGESTIONS:\n{suggestions_text}\n\n"
        f"DRAFT PROMPT TO CORRECT:\n{draft_prompt}\n\n"
        f"Task:\n"
        f"Rewrite the draft prompt to address all suggestions. For example:\n"
        f"- Convert any negative constraints (like 'no blur', 'no flickering') into positive statements (like 'crisp, stable motion').\n"
        f"- Remove any forbidden terms (like 'epic').\n"
        f"- Ensure it meets length requirements while preserving all original staging, actions, character garments, and Segmind reference wording (like image 1 or original clip reference images).\n"
        f"Return the refined prompt in your structured output."
    )
    
    try:
        response_json = generate_structured(
            provider=provider,
            client=client,
            model=model,
            contents=[refiner_prompt],
            schema=RefinedPromptResult,
            system_instruction=get_prompt_template(
                "prompt_refiner_system",
                "You are a professional prompt optimizer. You rewrite AI video prompts to resolve quality issues while keeping the narrative, references, and wardrobe details intact."
            ),
            temperature=0.1,
        )
        refined = response_json.get("english_prompt", "").strip()
        if refined:
            print("Auto-refinement completed successfully.")
            return refined
    except Exception as exc:
        print(f"Warning: Auto-refinement LLM request failed: {exc}")
        
    return draft_prompt


def generate_video_prompts_batch(
    client: genai.Client | OpenAI,
    clusters: List[Dict[str, Any]],
    reference_assets: List[str],
    assets_dir: str,
    batch_size: int = 5,
    prompt_skill_text: Optional[str] = None,
    provider: Optional[Provider] = None,
    model: Optional[str] = None,
    initial_frames_dir: Optional[str] = None,
    video_frames_dir: Optional[str] = None,
    run_validator: bool = True,
    generate_initial_frame: bool = True,
) -> List[Dict[str, Any]]:
    """Generate one prompt result per video cluster in bounded request batches."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not clusters:
        return []

    provider = provider or _detect_provider(client)
    model = model or _default_model_for_provider(provider)
    if initial_frames_dir is None:
        initial_frames_dir = os.path.join("data", "output", "initial_frames")
    if video_frames_dir is None:
        video_frames_dir = os.path.join("data", "output", "video_frames")

    indexed_clusters = list(enumerate(clusters))
    uploaded_refs = []
    reference_handles = {}
    clip_handles = {}
    cluster_frame_paths = {}
    initial_frame_image_paths = {}
    initial_frame_image_refs = {}
    selected_assets_by_id = {}
    results_by_id = {}

    try:
        for cluster_id, cluster in indexed_clusters:
            clip = cluster.get("matched_clip")
            clip_name = clip.get("clip") if clip else None
            if not clip_name:
                continue
            path = os.path.join(assets_dir, "06_clips", "_raw", clip_name)
            if not os.path.exists(path):
                continue
            frame_paths = media._extract_video_frames(
                path,
                os.path.join(video_frames_dir, _cluster_media_dir_name(cluster_id, cluster)),
                clip.get("duration_s", 0.0) if clip else 0.0,
            )
            if provider == "openai":
                frame_refs = [
                    _openai_file_reference(client, frame_path)
                    for frame_path in frame_paths
                ]
            else:
                frame_refs = [
                    upload._prepare_media_reference(client, frame_path, provider)
                    for frame_path in frame_paths
                ]
                frame_refs = [r for r in frame_refs if r is not None]
                uploaded_refs.extend(frame_refs)

            if frame_refs:
                clip_handles[cluster_id] = frame_refs
                cluster_frame_paths[cluster_id] = frame_paths

        for batch_start in range(0, len(indexed_clusters), batch_size):
            batch = indexed_clusters[batch_start:batch_start + batch_size]
            try:
                selected_assets_by_id.update(
                    _select_reference_assets_for_batch(
                        provider=provider,
                        client=client,
                        model=model,
                        batch=batch,
                        reference_assets=reference_assets,
                    )
                )
            except Exception as exc:
                print(f"Reference asset selection failed; continuing with clip frames only: {exc}")
                for cluster_id, _ in batch:
                    selected_assets_by_id[cluster_id] = []

            for path in dict.fromkeys(
                asset
                for cluster_id, _ in batch
                for asset in selected_assets_by_id.get(cluster_id, [])
            ):
                if path in reference_handles or not os.path.exists(path):
                    continue
                ref = upload._prepare_media_reference(client, path, provider)
                if ref:
                    reference_handles[path] = ref
                    uploaded_refs.append(ref)

            initial_frame_prompts = {}
            if generate_initial_frame:
                initial_frame_contents = [
                    (
                        "Generate the initial frame primarily from the extracted "
                        "original clip frames. Use selected character/location sheets "
                        "only to preserve identity, wardrobe, props, and named setting "
                        "details. Do not replace visible clip continuity with unrelated "
                        "details from an asset sheet."
                    )
                ]

                initial_frame_contents.append(
                    "Generate an initial frame prompt for every clip cluster below. "
                    "Each initial frame prompt must describe a single still image that "
                    "can be generated before video generation. Preserve the original "
                    "clip's visible continuity, selected character/location/prop sheets, "
                    "client feedback, frame size, and aspect ratio. Return every supplied "
                    "cluster_id exactly once."
                )
                batch_ids = set()
                for cluster_id, cluster in batch:
                    batch_ids.add(cluster_id)
                    initial_frame_contents.append(_cluster_prompt_context(cluster_id, cluster))
                    _append_selected_reference_assets(
                        initial_frame_contents,
                        selected_assets_by_id.get(cluster_id, []),
                        reference_handles,
                    )
                    _append_clip_reference(
                        initial_frame_contents,
                        assets_dir,
                        clip_handles,
                        cluster,
                        cluster_id,
                    )

                try:
                    response_json = generate_structured(
                        provider=provider,
                        client=client,
                        model=model,
                        contents=initial_frame_contents,
                        schema=BatchInitialFramePromptResult,
                        system_instruction=get_prompt_template(
                            "keyframe_prompt_engineer_system",
                            "You are a professional keyframe prompt engineer. "
                            "Write English-only prompts for initial still frames "
                            "that anchor a later image-to-video generation."
                        ),
                    )
                    for item in response_json.get("results", []):
                        cluster_id = item.get("cluster_id")
                        if cluster_id in batch_ids and cluster_id not in initial_frame_prompts:
                            initial_frame_prompts[cluster_id] = item.get(
                                "initial_frame_prompt", ""
                            ).strip()
                except Exception as exc:
                    explanation = f"Initial frame generation failed: {exc}"
                    for cluster_id, _ in batch:
                        results_by_id[cluster_id] = _failed_batch_result(
                            cluster_id, explanation
                        )
                    continue

                for cluster_id, _ in batch:
                    if not initial_frame_prompts.get(cluster_id):
                        results_by_id[cluster_id] = _failed_batch_result(
                            cluster_id,
                            "The model response omitted this initial frame cluster_id.",
                        )

                if provider == "openai":
                    for cluster_id, cluster in batch:
                        if cluster_id in results_by_id:
                            continue
                        try:
                            image_path = _generate_openai_initial_frame_image(
                                client=client,
                                prompt=initial_frame_prompts[cluster_id],
                                cluster_id=cluster_id,
                                cluster=cluster,
                                initial_frames_dir=initial_frames_dir,
                            )
                            initial_frame_image_paths[cluster_id] = image_path
                            initial_frame_image_refs[cluster_id] = _openai_file_reference(
                                client,
                                image_path,
                            )
                        except Exception as exc:
                            results_by_id[cluster_id] = _failed_batch_result(
                                cluster_id,
                                f"Initial frame image generation failed: {exc}",
                            )
            else:
                for cluster_id, _ in batch:
                    initial_frame_prompts[cluster_id] = ""

            prompt_batch = [
                item
                for item in batch
                if item[0] not in results_by_id
            ]
            if not prompt_batch:
                continue

            contents = [
                (
                    "Use the extracted original clip frames as the concrete visual reference. "
                    "Use selected character/location sheets only when they are listed for that "
                    "cluster, and keep their labels consistent with the supplied reference legend."
                )
                if not generate_initial_frame
                else (
                    "Use the extracted original clip frames and generated initial "
                    "frame image as the concrete visual anchor. Use selected "
                    "character/location sheets only when they are listed for that "
                    "cluster, and keep their labels consistent with the supplied "
                    "reference legend."
                )
            ]

            contents.append(
                "Generate one independent result for every clip cluster below. "
                "Write the final Seedance 2.0 video prompt from the selected image assets, "
                "original clip, and client feedback. Return every supplied cluster_id exactly once."
                if not generate_initial_frame
                else (
                    "Generate one independent result for every clip cluster below. "
                    "Use each INITIAL_FRAME_PROMPT as the first-frame visual anchor, "
                    "then write the final Seedance 2.0 video prompt from the initial "
                    "frame prompt, selected image assets, original clip, and feedback. "
                    "Return every supplied cluster_id exactly once."
                )
            )
            prompt_batch_ids = set()
            for cluster_id, cluster in prompt_batch:
                prompt_batch_ids.add(cluster_id)
                if generate_initial_frame:
                    contents.append(
                        f"{_cluster_prompt_context(cluster_id, cluster)}\n"
                        f"INITIAL_FRAME_PROMPT:\n{initial_frame_prompts[cluster_id]}"
                    )
                else:
                    contents.append(_cluster_prompt_context(cluster_id, cluster))
                selected_assets = selected_assets_by_id.get(cluster_id, [])
                contents.append(
                    "REFERENCE_LEGEND_TO_USE:\n"
                    f"{_reference_legend_for_assets(selected_assets)}"
                )
                _append_selected_reference_assets(
                    contents,
                    selected_assets,
                    reference_handles,
                )
                if generate_initial_frame and cluster_id in initial_frame_image_refs:
                    contents.append(
                        "GENERATED_INITIAL_FRAME_IMAGE: Use this generated still as "
                        "the concrete first-frame visual anchor for the final video prompt."
                    )
                    _append_media_reference(contents, initial_frame_image_refs[cluster_id])
                _append_clip_reference(contents, assets_dir, clip_handles, cluster, cluster_id)
            # Lazy load the video generation skill prompt and overrides only when needed
            video_prompt_skill_text = prompt_skill_text or _default_seedance_skill_text()
            video_system_instruction = video_prompt_skill_text + get_prompt_template(
                "project_overrides", DEFAULT_PROJECT_OVERRIDES
            )

            try:
                response_json = generate_structured(
                    provider=provider,
                    client=client,
                    model=model,
                    contents=contents,
                    schema=BatchPromptResult,
                    system_instruction=video_system_instruction,
                )
                for item in response_json.get("results", []):
                    cluster_id = item.get("cluster_id")
                    if cluster_id not in prompt_batch_ids or cluster_id in results_by_id:
                        continue
                    prompt = item.get("english_prompt", "").strip()
                    
                    # Run prompt quality checks
                    if run_validator:
                        cluster_dict = next(c for cid, c in prompt_batch if cid == cluster_id)
                        feedback_items = cluster_dict.get("feedback_items", [])
                        selected_assets = selected_assets_by_id.get(cluster_id, [])
                        has_clip = len(cluster_frame_paths.get(cluster_id, [])) > 0
                        
                        eval_model = _default_model_for_provider(provider)
                        print(f"Running quality checks for cluster {cluster_id}...")
                        quality_report = run_quality_check(
                            provider=provider,
                            client=client,
                            model=eval_model,
                            prompt=prompt,
                            feedback_items=feedback_items,
                            selected_assets=selected_assets,
                            has_clip=has_clip,
                        )
                        
                        # Loop to rewrite the prompt up to a maximum of 3 times if validator fails
                        rewrite_attempts = 0
                        max_rewrites = 3
                        while not quality_report.get("passed", False) and quality_report.get("suggestions") and rewrite_attempts < max_rewrites:
                            rewrite_attempts += 1
                            print(f"Cluster {cluster_id} failed quality checks. Triggering refinement iteration {rewrite_attempts} of {max_rewrites}...")
                            
                            prompt = _refine_prompt(
                                provider=provider,
                                client=client,
                                model=eval_model,
                                draft_prompt=prompt,
                                feedback_items=feedback_items,
                                suggestions=quality_report.get("suggestions", []),
                            )
                            
                            print(f"Running quality checks on refined prompt (iteration {rewrite_attempts}) for cluster {cluster_id}...")
                            quality_report = run_quality_check(
                                provider=provider,
                                client=client,
                                model=eval_model,
                                prompt=prompt,
                                feedback_items=feedback_items,
                                selected_assets=selected_assets,
                                has_clip=has_clip,
                            )
                        
                        status = "success"
                        quality_warning = None
                        if not quality_report.get("passed", False):
                            status = "warning"
                            suggestions = quality_report.get("suggestions", [])
                            if suggestions:
                                quality_warning = f"Quality check warning: {suggestions[0]}"
                                if len(suggestions) > 1:
                                    quality_warning += f" (and {len(suggestions)-1} other issues)"
                            else:
                                quality_warning = "Failed quality validation checks."
                    else:
                        selected_assets = selected_assets_by_id.get(cluster_id, [])
                        quality_report = None
                        quality_warning = None
                        status = "success"
                            
                    results_by_id[cluster_id] = {
                        "cluster_id": cluster_id,
                        "initial_frame_prompt": initial_frame_prompts[cluster_id],
                        "initial_frame_image_path": initial_frame_image_paths.get(cluster_id, ""),
                        "clip_frame_paths": cluster_frame_paths.get(cluster_id, []),
                        "selected_assets": selected_assets,
                        "prompt_format": item.get("prompt_format"),
                        "reference_legend": _reference_legend_for_assets(selected_assets),
                        "video_model_prompt": prompt,
                        "explanation": item.get("explanation"),
                        "status": status,
                        "quality_warning": quality_warning,
                        "quality_report": quality_report,
                    }
            except Exception as exc:
                explanation = f"Batch generation failed: {exc}"
                for cluster_id, _ in prompt_batch:
                    results_by_id[cluster_id] = _failed_batch_result(
                        cluster_id, explanation
                    )

            for cluster_id, _ in prompt_batch:
                if cluster_id not in results_by_id:
                    results_by_id[cluster_id] = _failed_batch_result(
                        cluster_id,
                        "The model response omitted this cluster_id.",
                    )

        return [results_by_id[cluster_id] for cluster_id, _ in indexed_clusters]
    finally:
        for ref in uploaded_refs:
            try:
                if provider == "openai" and isinstance(ref, OpenAIFileReference):
                    if ref.file_id:
                        client.files.delete(ref.file_id)
                else:
                    client.files.delete(name=ref.name)
            except Exception as exc:
                ref_name = getattr(ref, "name", getattr(ref, "file_id", "unknown"))
                print(f"Error deleting file {ref_name}: {exc}")


def generate_single_video_prompt(
    client: genai.Client | OpenAI,
    cluster: Dict[str, Any],
    reference_assets: List[str],
    assets_dir: str,
    prompt_skill_text: Optional[str] = None,
    provider: Optional[Provider] = None,
    model: Optional[str] = None,
    initial_frames_dir: Optional[str] = None,
    video_frames_dir: Optional[str] = None,
    run_validator: bool = True,
    generate_initial_frame: bool = True,
) -> Dict[str, Any]:
    """Generate one two-pass video prompt without processing additional clusters."""
    results = generate_video_prompts_batch(
        client=client,
        clusters=[cluster],
        reference_assets=reference_assets,
        assets_dir=assets_dir,
        batch_size=1,
        prompt_skill_text=prompt_skill_text,
        provider=provider,
        model=model,
        initial_frames_dir=initial_frames_dir,
        video_frames_dir=video_frames_dir,
        run_validator=run_validator,
        generate_initial_frame=generate_initial_frame,
    )
    if not results:
        return _failed_batch_result(0, "No result generated for the supplied cluster.")
    return results[0]


def generate_video_prompt(
    client: genai.Client,
    remark: str,
    clip_name: Optional[str],
    clip_duration_s: float,
    start_tc: str,
    end_tc: str,
    selected_assets: List[str],
    raw_clip_path: Optional[str]
) -> Dict[str, Any]:
    """Uploads context sheets and raw clip, then calls REASONING_MODEL
    to generate a cinematic prompt for AI video generation.

    Args:
        client (genai.Client): The Google GenAI client instance used to interact with the model.
        remark (str): The feedback remark provided by the client, which may contain references to characters or locations.
        clip_name (Optional[str]): The name of the video clip being modified, or None if unknown.
        clip_duration_s (float): The duration of the video clip in seconds.
        start_tc (str): The start timecode of the clip segment in MM:SS or H:MM:SS format.
        end_tc (str): The end timecode of the clip segment in MM:SS or H:MM:SS format.
        selected_assets (List[str]): A list of selected asset file paths that are relevant based on the feedback.
        raw_clip_path (Optional[str]): The local path to the raw video clip file, or None if not available.

    Returns:
        Dict[str, Any]: A dictionary containing the generated video model prompt, an explanation, and a status indicating success or failure.
    """
    uploaded_refs = []
    try:
        # 1. Upload dynamically selected assets (character/location sheets)
        uploaded_assets_refs = []
        for asset_path in selected_assets[:3]:  # Limit to 3 files to avoid bloating context
            ref = upload.upload_file_and_wait(client, asset_path)
            if ref:
                uploaded_assets_refs.append(ref)
                uploaded_refs.append(ref)
                
        # 2. Upload raw video clip if it exists
        raw_clip_ref = None
        if raw_clip_path and os.path.exists(raw_clip_path):
            raw_clip_ref = upload.upload_file_and_wait(client, raw_clip_path)
            if raw_clip_ref:
                uploaded_refs.append(raw_clip_ref)
                
        print(f"Assets ready. Generating multimodal video model prompt using {REASONING_MODEL}...")
        
        # 3. Construct the prompt package
        contents = []
        if raw_clip_ref:
            contents.append(raw_clip_ref)
        contents.extend(uploaded_assets_refs)
        
        prompt_text = (
            f"You are modifying the video clip: \"{clip_name or 'unknown'}\" "
            f"(Duration: {clip_duration_s}s, Segment: {start_tc} to {end_tc}).\n\n"
            f"Client Feedback Remark: \"{remark}\"\n\n"
            f"Tasks:\n"
            f"1. Analyze the original clip's action, framing, lighting, and camera motion.\n"
            f"2. Analyze the provided preproduction reference sheets (like character details or locations) to ensure design consistency in the generated prompt.\n"
            f"3. Generate a highly descriptive text-to-video / image-to-video prompt (suitable for models like Runway Gen-3, Sora, or Luma Dream Machine) that creates the revised shot incorporating the client's feedback while maintaining absolute consistency with the preproduction sheets.\n\n"
            f"Please generate and return the prompt."
        )
        contents.append(prompt_text)
        
        # 4. Invoke the model
        response = client.models.generate_content(
            model=REASONING_MODEL,  # Flash supports multimodal file inputs
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=PromptResult,
                temperature=0.2,
                system_instruction=get_prompt_template(
                    "director_prompt_engineer_system",
                    "You are a professional film director and AI video prompt engineer. "
                    "You generate extremely detailed, consistent cinematic prompts for video generation models, "
                    "ensuring client edits are precisely followed and preproduction design assets are visually respected."
                )
            )
        )
        
        response_json = json.loads(response.text)
        return {
            "video_model_prompt": response_json.get("video_model_prompt"),
            "explanation": response_json.get("explanation"),
            "status": "success"
        }
        
    except Exception as e:
        print(f"Error in video prompt generator: {e}")
        return {
            "video_model_prompt": f"Failed to generate: {e}",
            "explanation": str(e),
            "status": "failed"
        }
        
    finally:
        # 5. Clean up File API uploads immediately to avoid storage clutter
        if uploaded_refs:
            print("Cleaning up files from Gemini File API...")
            for ref in uploaded_refs:
                try:
                    client.files.delete(name=ref.name)
                    print(f"Deleted {ref.name}")
                except Exception as ex:
                    print(f"Error deleting file {ref.name}: {ex}")


def generate_audio_instruction(
    client: genai.Client,
    remark: str,
    start_tc: str,
    end_tc: str
) -> Dict[str, Any]:
    """
    Calls LITE_MODEL (high rate limit, text-only)
    to generate post-production audio instructions.
    """
    try:
        print(f"Generating audio instruction using {LITE_MODEL}...")
        prompt = (
            f"A client has provided audio feedback for the segment from {start_tc} to {end_tc}.\n"
            f"Feedback: \"{remark}\"\n\n"
            f"Generate a clear, structured instruction prompt for a voiceover artist, sound editor, or SFX generator "
            f"to address this feedback."
        )
        
        response = client.models.generate_content(
            model=LITE_MODEL,  # Text-only optimized model
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=AudioInstructionResult,
                temperature=0.2,
                system_instruction=get_prompt_template(
                    "audio_supervisor_system",
                    "You are a precise post-production supervisor formatting audio changes into production-ready instructions."
                )
            )
        )
        
        response_json = json.loads(response.text)
        return {
            "audio_instruction": response_json.get("audio_instruction"),
            "explanation": response_json.get("explanation"),
            "status": "success"
        }
        
    except Exception as e:
        print(f"Error in audio instruction generator: {e}")
        return {
            "audio_instruction": f"Failed to generate: {e}",
            "explanation": str(e),
            "status": "failed"
        }
