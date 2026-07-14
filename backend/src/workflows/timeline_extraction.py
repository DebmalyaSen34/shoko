import os
import sys
# Add project root to sys.path if running directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.parser import parse_prproj_to_json

def extract_timeline_from_project(
    prproj_path: str,
    project_name: str,
    output_base_dir: str = "data"
) -> str:
    """
    Extracts timeline data from an Adobe Premiere Pro project (.prproj) file
    and saves it as timeline.json in a project-specific directory.

    Args:
        prproj_path: Path to the .prproj file.
        project_name: Name of the project.
        output_base_dir: Base directory where timelines are stored. Defaults to "data".

    Returns:
        The absolute path to the generated timeline.json file.

    Raises:
        FileNotFoundError: If the prproj_path does not exist.
        ValueError: If the file is not a valid .prproj file.
    """
    abs_prproj_path = os.path.abspath(prproj_path)
    if not os.path.exists(abs_prproj_path):
        raise FileNotFoundError(f"Project file not found at: {abs_prproj_path}")

    # Build target output file path
    abs_output_base_dir = os.path.abspath(output_base_dir)
    target_dir = os.path.join(abs_output_base_dir, project_name)
    os.makedirs(target_dir, exist_ok=True)
    
    output_json_path = os.path.join(target_dir, "timeline.json")

    # Use the parser module to parse and write the timeline json
    parse_prproj_to_json(abs_prproj_path, output_json_path)

    return os.path.abspath(output_json_path)

if __name__ == "__main__":
    prproj_path = "project-red-and-green-premier-pro/project_red_and_green_complete.prproj"
    project_name = "project-red-and-green-ep5"

    output_json_path = extract_timeline_from_project(prproj_path, project_name)
    print(f"Timeline extracted and saved to: {output_json_path}")
