import os

import pytest
from dotenv import load_dotenv
from supabase import create_client


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_SUPABASE_INTEGRATION_TESTS") != "1",
    reason="Supabase integration test is opt-in. Set RUN_SUPABASE_INTEGRATION_TESTS=1 to enable.",
)


def test_supabase_storage_bucket_is_reachable(tmp_path):
    load_dotenv()

    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = (
        os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        or os.getenv("SUPABASE_KEY")
        or os.getenv("SUPABASE_ANON_KEY")
    )
    bucket = os.getenv("SUPABASE_AUDIO_BUCKET", "seedance-audio")

    if not supabase_url or not supabase_key:
        pytest.skip("SUPABASE_URL and a Supabase key are required for this integration test.")

    client = create_client(supabase_url, supabase_key)
    payload = tmp_path / "supabase-test.txt"
    payload.write_text("Supabase Storage Test", encoding="utf-8")
    storage_path = "test/supabase-test.txt"

    with payload.open("rb") as file:
        result = client.storage.from_(bucket).upload(
            storage_path,
            file,
            {"content-type": "text/plain", "upsert": "true"},
        )

    public_url = client.storage.from_(bucket).get_public_url(storage_path)

    assert result is not None
    assert storage_path in public_url
