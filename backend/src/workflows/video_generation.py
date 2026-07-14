import os
import sys
# Add project root to sys.path if running directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
from typing import List, Optional

from scripts.generate_seedance_video import (
    build_seedance_content,
    clamp_duration,
    create_seedance_task,
    poll_seedance_task,
    download_video,
    _response_value,
    _video_url_from_task
)

def run_video_generation_workflow(
    prompts_json_path: str,
    project_name: str,
    ark_api_key: Optional[str] = None,
    output_dir: Optional[str] = None,
    poll_interval_seconds: int = 15,
    timeout_seconds: int = 1800,
    watermark: bool = False
) -> List[str]:
    """
    Executes the video generation tasks on BytePlus ModelArk (Seedance 2.0).
    Loads prompts, creates tasks, polls for completion, downloads videos,
    and moves final files to assets/{project_name}/06_clips/_final/{clip_used}.
    """
    api_key = ark_api_key or os.environ.get("ARK_API_KEY")
    if not api_key:
        raise ValueError("ARK_API_KEY is not set. Please supply it or define the environment variable.")

    abs_prompts_path = os.path.abspath(prompts_json_path)
    if not os.path.exists(abs_prompts_path):
        raise FileNotFoundError(f"Prompts JSON not found at: {abs_prompts_path}")

    with open(abs_prompts_path, 'r', encoding='utf-8') as f:
        prompt_items = json.load(f)

    if isinstance(prompt_items, dict):
        prompt_items = [prompt_items]

    # Output directory for final clips inside project workspace
    final_clips_dir = os.path.abspath(os.path.join("assets", project_name, "06_clips", "_final"))
    os.makedirs(final_clips_dir, exist_ok=True)

    # Base folder for raw downloads
    resolved_output_dir = output_dir or os.path.join("data", project_name, "seedance_videos")
    os.makedirs(resolved_output_dir, exist_ok=True)

    generated_clip_paths = []

    for item in prompt_items:
        clip_name = item.get("clip_used")
        generation_type = item.get("generation_type", "none")
        prompt_text = item.get("video_model_prompt")

        # Skip segments with generation_type none or missing prompt
        if generation_type == "none" or not clip_name or not prompt_text:
            print(f"Skipping segment for {clip_name or 'unknown'} (generation_type: {generation_type})")
            continue

        print(f"Processing video generation for clip: {clip_name}...")

        # 1. Build content payload (incorporating frames, assets, initial frames, audio)
        content_payload = build_seedance_content(
            item=item,
            use_local_initial_frame=True,
            initial_image_url=None
        )

        model = os.environ.get("SEEDANCE_MODEL", "dreamina-seedance-2-0-260128")
        duration = clamp_duration(item.get("duration", 5), model)
        ratio = item.get("ratio", "9:16")
        generate_audio = item.get("generate_audio", False)

        print(f"Creating task on ModelArk (model: {model}, generate_audio: {generate_audio})...")

        # 2. Submit task
        create_result = create_seedance_task(
            api_key=api_key,
            model=model,
            content=content_payload,
            ratio=ratio,
            duration=duration,
            generate_audio=generate_audio,
            watermark=watermark
        )

        task_id = _response_value(create_result, "id")
        if not task_id:
            raise RuntimeError(f"Failed to retrieve task ID from creation response: {create_result}")
        print(f"Task created successfully. Task ID: {task_id}")

        # 3. Poll task status
        print(f"Polling task {task_id} (interval: {poll_interval_seconds}s, timeout: {timeout_seconds}s)...")
        task_info = poll_seedance_task(
            api_key=api_key,
            task_id=task_id,
            poll_interval_seconds=poll_interval_seconds,
            timeout_seconds=timeout_seconds
        )

        # 4. Extract download URL
        video_url = _video_url_from_task(task_info)
        if not video_url:
            raise RuntimeError(f"Seedance task succeeded but did not return video_url: {task_info}")

        # 5. Download and move to final destination
        temp_download_path = os.path.join(resolved_output_dir, f"{task_id}.mp4")
        download_video(video_url, temp_download_path)

        final_clip_path = os.path.join(final_clips_dir, clip_name)
        os.rename(temp_download_path, final_clip_path)

        print(f"Generated clip successfully saved to: {final_clip_path}")
        generated_clip_paths.append(os.path.abspath(final_clip_path))

    return generated_clip_paths

if __name__ == "__main__":
    import argparse

    from dotenv import load_dotenv

    load_dotenv()  # Load environment variables from .env file if present

    parser = argparse.ArgumentParser(description="Run video generation workflow for Seedance 2.0.")
    parser.add_argument("prompts_json_path", type=str, help="Path to the prompts JSON file.")
    parser.add_argument("project_name", type=str, help="Project name for organizing output clips.")
    parser.add_argument("--ark_api_key", type=str, default=os.environ.get("ARK_API_KEY"), help="Optional ARK API key (overrides env variable).")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory for temporary downloads. Defaults to data/{project_name}/seedance_videos.")
    parser.add_argument("--poll_interval_seconds", type=int, default=15, help="Polling interval in seconds.")
    parser.add_argument("--timeout_seconds", type=int, default=1800, help="Timeout for task completion in seconds.")
    parser.add_argument("--watermark", action='store_true', help="Whether to apply watermark to generated videos.")

    args = parser.parse_args()

    generated_clips = run_video_generation_workflow(
        prompts_json_path=args.prompts_json_path,
        project_name=args.project_name,
        ark_api_key=args.ark_api_key,
        output_dir=args.output_dir,
        poll_interval_seconds=args.poll_interval_seconds,
        timeout_seconds=args.timeout_seconds,
        watermark=args.watermark
    )

    print(f"Video generation workflow completed. Generated clips: {generated_clips}")