import os
import sys
import json
from google import genai
from google.genai import types
from dotenv import load_dotenv

from config.settings import LITE_MODEL
from .schemas import FeedbackList
from .utils import get_prompt_template


load_dotenv()

def process_feedback(input_file: str, output_file: str):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable is not set.", file=sys.stderr)
        print("Please set it in your terminal before running this script:", file=sys.stderr)
        print("  export GEMINI_API_KEY='your-api-key-here'", file=sys.stderr)
        sys.exit(1)
        
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.", file=sys.stderr)
        sys.exit(1)

    with open(input_file, 'r', encoding='utf-8') as f:
        feedback_content = f.read()

    print(f"Initializing Gemini client and processing '{input_file}' using {LITE_MODEL}...")
    
    # Initialize the Gemini GenAI client
    client = genai.Client(api_key=api_key)
    
    # Create the analysis prompt
    prompt = (
        "You are an expert video post-production supervisor.\n"
        "Analyze the following feedback transcript line-by-line.\n"
        "For each line or distinct remark, extract the timestamp if present, "
        "and categorize it into 'audio', 'video', or 'both'.\n\n"
        "Classification Rules:\n"
        "- 'audio': Dubbing issues, pronunciation errors, sound effects (sfx), dialogues, background noise/music.\n"
        "- 'video': Camera angles, visual reaction changes, speed/duration adjustments, facial expressions, physical action/motion.\n"
        "- 'both': If the feedback involves a combination of both audio and video changes at that timestamp.\n\n"
        "Feedback Transcript:\n"
        f"{feedback_content}"
    )

    try:
        # Call Gemini API with structured JSON output enforced by Pydantic schema
        response = client.models.generate_content(
            model=LITE_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=FeedbackList,
                temperature=0.1,
                system_instruction=get_prompt_template(
                    "feedback_categorizer_system",
                    "You are a precise data extractor that parses unstructured video editor feedback text "
                    "into a structured list of remarks matching the schema exactly."
                )
            ),
        )
        
        # Load the structured JSON response
        response_data = json.loads(response.text)
        remarks = response_data.get("remarks", [])
        
        # Write the resulting list to feedback.json
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(remarks, f, indent=2, ensure_ascii=False)
            
        print(f"Successfully processed feedback! Output written to '{output_file}'")
        print(f"Total remarks categorized: {len(remarks)}")
        
    except Exception as e:
        print(f"An error occurred while calling the Gemini API: {e}", file=sys.stderr)
        sys.exit(1)
