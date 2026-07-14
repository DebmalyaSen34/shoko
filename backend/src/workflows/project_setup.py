import os
import shutil
import tempfile
import zipfile
from typing import Tuple, Optional

# Define recognized file extensions (case-insensitive)
VIDEO_EXTENSIONS = {
    ".mp4", ".mov", ".avi", ".mxf", ".r3d", ".mkv", ".webm", ".flv", ".wmv"
}

AUDIO_EXTENSIONS = {
    ".mp3", ".wav", ".m4a", ".aac", ".flac", ".aif", ".aiff", ".ogg"
}

def setup_project_workspace(
    zip_path: str,
    project_name: str,
    assets_dir: str = "assets"
) -> Tuple[str, str]:
    """
    Extracts the Adobe Premiere Pro project zip file, creates a standard directory structure
    for the project within assets_dir, and organizes video, audio, and .prproj files.

    Args:
        zip_path: Path to the Adobe Premiere Pro project zip file from the user.
        project_name: Name of the project (folder name to be created).
        assets_dir: The directory under which the project_name folder will be created. Defaults to "assets".

    Returns:
        A tuple: (project_dir_path, prproj_file_path)
            project_dir_path: Absolute path to the created project directory.
            prproj_file_path: Absolute path to the moved .prproj file.

    Raises:
        FileNotFoundError: If the zip_path does not exist.
        ValueError: If no .prproj file is found in the zip archive.
    """
    # 1. Resolve paths
    abs_zip_path = os.path.abspath(zip_path)
    if not os.path.exists(abs_zip_path):
        raise FileNotFoundError(f"Zip file not found at: {abs_zip_path}")

    abs_assets_dir = os.path.abspath(assets_dir)
    project_dir = os.path.join(abs_assets_dir, project_name)

    # 2. Create directory structure
    subdirs = [
        "00_style",
        "01_characters",
        "02_props",
        "03_locations",
        "04_audio",
        "05_references",
        os.path.join("06_clips", "_final"),
        os.path.join("06_clips", "_raw")
    ]

    for subdir in subdirs:
        target_path = os.path.join(project_dir, subdir)
        os.makedirs(target_path, exist_ok=True)

    # Destination paths for files
    audio_dest_dir = os.path.join(project_dir, "04_audio")
    clips_raw_dest_dir = os.path.join(project_dir, "06_clips", "_raw")

    prproj_file_path: Optional[str] = None

    # 3. Extract the zip file in a temp location
    with tempfile.TemporaryDirectory() as temp_dir:
        # Extract files
        with zipfile.ZipFile(abs_zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        # 4. Walk through the temporary location and organize files
        for root, _, files in os.walk(temp_dir):
            for file in files:
                # Skip OS specific metadata files (e.g. .DS_Store, __MACOSX)
                if file.startswith('.') or file.startswith('__MACOSX'):
                    continue

                file_path = os.path.join(root, file)
                _, ext = os.path.splitext(file)
                ext_lower = ext.lower()

                if ext_lower == ".prproj":
                    # Move .prproj to the project_name directory
                    dest_path = os.path.join(project_dir, file)
                    shutil.move(file_path, dest_path)
                    # Keep track of the first prproj file path found
                    if not prproj_file_path:
                        prproj_file_path = dest_path
                elif ext_lower in VIDEO_EXTENSIONS:
                    # Move video file to 06_clips/_raw
                    shutil.move(file_path, os.path.join(clips_raw_dest_dir, file))
                elif ext_lower in AUDIO_EXTENSIONS:
                    # Move audio file to 04_audio
                    shutil.move(file_path, os.path.join(audio_dest_dir, file))

    # 5. Verify a .prproj file was found
    if not prproj_file_path:
        raise ValueError(f"No .prproj file found in the zip archive: {zip_path}")

    # 6. Return paths
    return os.path.abspath(project_dir), os.path.abspath(prproj_file_path)
