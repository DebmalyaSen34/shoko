import os
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()

def cal_tokens(prompt):
    import tiktoken
    enc = tiktoken.encoding_for_model("gpt-5")
    tokens = enc.encode(prompt)
    return len(tokens)

def transcribe_audio(file_path):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    with open(file_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="gpt-4o-transcribe",
            file=audio_file
        )
    return transcript.text

def gen_seedream_image():
    client = OpenAI( 
        base_url="https://ark.ap-southeast.bytepluses.com/api/v3", 
        api_key=os.getenv('ARK_API_KEY'), 
    )
    imagesResponse = client.images.generate( 
        model="seedream-5-0-260128",
        prompt="Keep the model's pose and the flowing shape of the liquid dress unchanged. Change the clothing material from silver metal to completely transparent clear water (or glass). Through the liquid water, the model's skin details are visible. Lighting changes from reflection to refraction.",
        size="2K",
        output_format="png",
        response_format="url",
        extra_body = {
            "image": "https://ark-doc.tos-ap-southeast-1.bytepluses.com/doc_image/seedream4_5_imageToimage.png",
            "watermark": False
        }
    ) 



if __name__ == "__main__":
    import json
    with open("data/project-red-and-green/video_prompts.json", "r") as f:
        prompts = json.load(f)

    prompt = [p for p in prompts if p["clip_used"]=="4-output (3).mp4"]

    with open("data/project-red-and-green/video_prompts_short.json", "w") as f:
        json.dump(prompt, f, indent=4)