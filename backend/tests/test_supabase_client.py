from supabase import create_client, Client
import tempfile
import os

from dotenv import load_dotenv

load_dotenv()

# Replace these with your credentials
SUPABASE_URL = os.getenv("SUPABASE_URL")  # e.g., "https://xyzcompany.supabase.co"
SUPABASE_KEY = os.getenv("SUPABASE_KEY")  # e.g., "your-anon-or-service-role-key"

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in the environment variables.")

BUCKET = "seedance-audio"  # Change to your bucket name

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

print(supabase.storage.from_(BUCKET).list())

# Create a temporary test file
with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
    f.write(b"Supabase Storage Test")
    file_path = f.name

storage_path = "test/supabase-test.txt"

try:
    # Upload
    with open(file_path, "rb") as f:
        result = supabase.storage.from_(BUCKET).upload(
            storage_path,
            f,
            {"content-type": "text/plain"}
        )

    print("✅ Upload successful")
    print(result)

    # Get public URL
    public_url = supabase.storage.from_(BUCKET).get_public_url(storage_path)
    print("\nPublic URL:")
    print(public_url)

except Exception as e:
    print("❌ Error:")
    print(e)

finally:
    os.remove(file_path)
