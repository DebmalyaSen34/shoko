import os
import json
from typing import List
from google.genai import types
from .schemas import SelectedAssets
from config.settings import LITE_MODEL, REASONING_MODEL
from .utils import get_prompt_template


#NOTE: for now this is local but later a cloud storage will be used, so the paths will be relative to the project root directory. The function will return relative paths to the project root.

def scan_assets(directory: str) -> List[str]:
    """Recursively list all files in the assets directory, excluding system files.

    Args:
        directory (str): The root directory to scan for asset files.

    Returns:
        List[str]: A list of relative file paths for all valid asset files found in the directory and its subdirectories.
    """
    
    asset_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.startswith('.'):
                continue
            full_path = os.path.join(root, file)
            # Use relative path from the project root directory
            rel_path = os.path.relpath(full_path, start="/Users/debmalyasen/developement/loka")
            asset_files.append(rel_path)
    return asset_files


def scan_visual_reference_assets(directory: str) -> List[str]:
    """Return character and location reference files without a model call."""
    references = []
    for subdirectory in ("01_characters", "03_locations"):
        root_directory = os.path.join(directory, subdirectory)
        for root, _, files in os.walk(root_directory):
            for filename in files:
                if filename.startswith("."):
                    continue
                references.append(os.path.join(root, filename))
    return sorted(references)

def select_assets_dynamically(client, remark: str, all_assets: List[str]) -> List[str]:
    """Use LITE_MODEL to select relevant character and location sheets dynamically.

    Args:
        client (_type_): The Google GenAI client instance used to interact with the model.
        remark (str): The feedback remark provided by the client, which may contain references to characters or locations.
        all_assets (List[str]): A list of all available asset file paths.

    Returns:
        List[str]: A list of selected asset file paths that are relevant based on the feedback.
    """

    assets_list_str = "\n".join([f"- {path}" for path in all_assets])
    
    prompt = (
        f"You are a post-production supervisor. We have a set of preproduction design assets (character reference sheets, locations, etc.):\n"
        f"{assets_list_str}\n\n"
        f"We received the following client feedback remark on a video clip:\n"
        f"\"{remark}\"\n\n"
        f"Analyze this remark. Identify which character sheets, location sheets, or visual references from the list are relevant context files. "
        f"Note:\n"
        f"- Do NOT select raw video clips (like files in 06_clips/_raw) or audio files (like files in 04_audio).\n"
        f"- Select only character reference sheets (01_characters) and location sheets (03_locations) that match who and where is mentioned in the feedback.\n"
        f"Return the selected relative file paths."
    )
    
    print(f"Querying {LITE_MODEL} for dynamic asset selection...")
    response = client.models.generate_content(
        model=LITE_MODEL, # Using higher rate limit model for text-only task
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=SelectedAssets,
            temperature=0.1,
            system_instruction=get_prompt_template(
                "asset_selector_system",
                "You are a precise tool that filters a file list based on feedback context."
            )
        )
    )
    
    response_json = json.loads(response.text)
    selected = response_json.get("relevant_assets", [])
    print(f"Asset Selection Reasoning: {response_json.get('reasoning')}")
    print(f"Selected Assets: {selected}")
    
    # Verify file paths exist
    valid_selected = []
    for p in selected:
        clean_p = p.strip()
        if os.path.exists(clean_p):
            valid_selected.append(clean_p)
        else:
            # Check absolute path
            abs_p = os.path.abspath(clean_p)
            if os.path.exists(abs_p):
                valid_selected.append(abs_p)
            else:
                print(f"Warning: Selected asset {clean_p} does not exist. Skipping.")
    return valid_selected
