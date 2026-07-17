import os
from segmind import SegmindClient, InferenceFailed, InferenceTimeout
from dotenv import load_dotenv
import base64
import mimetypes
from pathlib import Path
import requests

load_dotenv()

def to_data_url(path: str) -> str:
    file_path = Path(path)
    mime_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(file_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"

def upload_to_segmind(file_path: str) -> str:
    url = "https://workflows-api.segmind.com/upload-asset"
    api_key = os.getenv("SEGMIND_API_KEY")

    with open(file_path, "rb") as f:
        encoded_string = base64.b64encode(f.read()).decode("utf-8")
        data_url = f"data:{mimetypes.guess_type(file_path)[0]};base64,{encoded_string}"
    data = {
        "data_urls": [data_url]
    }

    headers = {
        'x-api-key': api_key,
        'accept': 'application/json, text/plain, */*',
        'content-type': 'application/json'
    }

    print(f"Uploading {file_path} to Segmind...")

    response = requests.post(url, json=data, headers=headers)
    result = response.json()

    asset_url = result["file_urls"][0]
    print(f"Uploaded {file_path} to Segmind. Asset URL: {asset_url}")
    return asset_url

client = SegmindClient(api_key=os.getenv("SEGMIND_API_KEY"))

prompt = """
Style & Mood: intimate domestic confrontation, warm beige interior, soft practical wall sconce, woven wall hanging, shallow depth of field, close vertical framing, quiet tension. Use image 1 as the first-frame visual anchor and preserve the exact room geometry, the blurred foreground shoulder at the frame edge, and the warm, lived-in lighting. Cross-check the room layout, lighting mood, and domestic details against image 2, image 3, image 4 and image 5 as supporting visual references.\n\nAvni, the mother, matches image 6 in average height, slim-to-average build, wavy shoulder-length brown hair, natural skin texture, and a restrained but annoyed presence. Keep Avni and Vir in the screen-used wardrobe from image 1 exactly, with @image4 and @image2 guiding face, hair, and character look. Start with Avni, the mother, looking around the room first, eyes moving off-camera as she scans for someone. Let frustration build before any dialogue: brow tightens, jaw sets, lips compress, breath controlled, irritation visible in her face and posture. Keep her body mostly still so the emotion reads through the face and shoulders. For Vir, use image 7 as the look reference. After Avni’s frustrated look-around and before her first line of dialogue, cut briefly to Vir’s image 8 evil smile reaction in a tight close-up, then back to Avni, the mother, with her irritation sharpened by what she has just seen. Maintain the same warm beige palette, soft shadow falloff, natural skin texture, and fluid motion throughout the cut. Render rich detail, tactile cinematic textures, a smooth, steady camera capture, and elegant continuity from shot to shot. Let the spoken exchange land through performance and framing, with the frame devoted to the scene.
"""

image_paths = [
    "/Users/debmalyasen/developement/loka/data/project-red-and-green/clip_frames/4-output (3)/frame_000.jpg",
    "/Users/debmalyasen/developement/loka/data/project-red-and-green/clip_frames/4-output (3)/frame_001.jpg",
    "/Users/debmalyasen/developement/loka/data/project-red-and-green/clip_frames/4-output (3)/frame_002.jpg",
    "/Users/debmalyasen/developement/loka/data/project-red-and-green/clip_frames/4-output (3)/frame_003.jpg",
    "/Users/debmalyasen/developement/loka/data/project-red-and-green/clip_frames/4-output (3)/frame_004.jpg",
    "/Users/debmalyasen/developement/loka/assets/project-red-and-green/01_characters/avni-sheet-v5.jpeg.jpg",
    "/Users/debmalyasen/developement/loka/assets/project-red-and-green/01_characters/vir-sheet-v1.png",
    "/Users/debmalyasen/developement/loka/data/project-red-and-green/referenced_frames/ref_00_44_output (22).jpg"
    ]

audio_path = "/Users/debmalyasen/developement/loka/assets/project-red-and-green/04_audio/RED GREEN BOYS EP 1(V2) - MIX.wav"

# image_urls = [upload_to_segmind(path) for path in image_paths]
image_urls = ['https://images.segmind.com/assets/deb@loka15.com/images/6047f893-ed80-4ceb-ac95-378965ba7e37.jpg', 'https://images.segmind.com/assets/deb@loka15.com/images/56e532e7-4548-4f02-a495-e13616969186.jpg', 'https://images.segmind.com/assets/deb@loka15.com/images/63d23f47-7226-44c1-959a-d0e799c18f3c.jpg', 'https://images.segmind.com/assets/deb@loka15.com/images/58b59507-cfff-4375-ae76-caec93717e65.jpg', 'https://images.segmind.com/assets/deb@loka15.com/images/38d4b0bc-b72e-42ae-ada7-469544d305f5.jpg', 'https://images.segmind.com/assets/deb@loka15.com/images/9f222f4a-e257-4d18-8a78-6a6764a5c614.jpg', 'https://images.segmind.com/assets/deb@loka15.com/images/90cd9ef0-b270-4fd4-a0cd-27590a55b7a3.png', 'https://images.segmind.com/assets/deb@loka15.com/images/eab98c7e-8134-473a-bb0d-8b80c71026b9.jpg']
print(f"Uploaded images to Segmind. Asset URLs: {image_urls}")

if image_urls:
    payload = {
        "prompt": prompt,
        "duration": 5,
        "resolution": "720p",
        "aspect_ratio": "9:16",
        "generate_audio": False,
        "reference_images": image_urls,
        "skip_moderation": True,
        "reference_audios": ["https://simple-unfeeling-refurnish.ngrok-free.dev/trimmed_RED%20GREEN%20BOYS%20EP%201%28V2%29%20-%20MIX.mp3"],
    }

    job = client.submit_async("seedance-2.0", **payload)
    print(job.request_id)

    try:
        result = job.wait(timeout=900, interval=2.0)
        print(result["status"])
        print(result.get("output"))
    except InferenceTimeout as e:
        print("still running: ", e.request_id)
    except InferenceFailed as e:
        print("failed: ", e.detail)
else:
    print("Failed to upload audio or images to Segmind.")