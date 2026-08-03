#!/usr/bin/env python3
"""Route smoke test for the server.py -> src/routers refactor.

Boots the FastAPI app in-process against a THROWAWAY storage directory
(`backend/.smoke_storage`, wiped at the start of every run so every run starts
from identical state), exercises every route (LLM calls and subprocesses
replaced by deterministic fakes), and records {status, body} per route.

Usage:
    python scripts/smoke_refactor.py baseline   # record scripts/smoke_baseline.json
    python scripts/smoke_refactor.py            # run, diff against baseline, exit 1 on diffs

Run `baseline` once against the untouched server.py, then re-run after every
cutover step. Any diff between the baseline and a later run signals a wiring
regression. POST /shutdown is intentionally skipped (it kills the process).
"""

import asyncio
import json
import os
import re
import shutil
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

STORAGE_DIR = BACKEND_DIR / ".smoke_storage"
BASELINE_PATH = Path(__file__).resolve().parent / "smoke_baseline.json"

FIXTURE_PROJECT = "smoke_project"
CREATED_PROJECT = "smoke_created_project"

# Run-varying values that must not cause false diffs between runs.
TS_RE = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?(?:Z|[+-]\d{2}:?\d{2})?")
UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
# Content hashes (16-hex) and generated version ids embed run-varying
# timestamps, so only their shape can be compared across runs.
HASH16_RE = re.compile(r"^[0-9a-f]{16}$")
VERSION_ID_RE = re.compile(r"^version_[0-9a-f]{12}$")


def normalize(value):
    """Replace run-varying values (timestamps, uuids, content hashes, file
    mtimes) so baseline and post-cutover runs compare cleanly. source_mtimes
    depend on when the fixture files were written, so only their keys matter,
    not their values."""
    if isinstance(value, dict):
        if "source_mtimes" in value and isinstance(value["source_mtimes"], dict):
            stripped = {k: v for k, v in value.items() if k != "source_mtimes"}
            return {
                **normalize(stripped),
                "source_mtimes": {k: "<mtime>" for k in value["source_mtimes"]},
            }
        return {k: normalize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [normalize(v) for v in value]
    if isinstance(value, str):
        value = UUID_RE.sub("<uuid>", TS_RE.sub("<ts>", value))
        if HASH16_RE.match(value):
            return "<hash>"
        if VERSION_ID_RE.match(value):
            return "<version_id>"
        return value
    return value


def build_fixture():
    """Create the throwaway storage dir with a synthetic project."""
    if STORAGE_DIR.exists():
        shutil.rmtree(STORAGE_DIR)
    assets_dir = STORAGE_DIR / "assets" / FIXTURE_PROJECT
    for subdir in ("00_style", "01_characters", "02_props", "03_locations",
                   "04_audio", "05_references", "06_clips/_final", "06_clips/_raw"):
        (assets_dir / subdir).mkdir(parents=True)

    project_dir = STORAGE_DIR / "data" / FIXTURE_PROJECT
    project_dir.mkdir(parents=True)

    timeline = {
        "sequence_name": "Smoke Sequence",
        "total_duration_tc": "00:00:10:00",
        "total_duration_s": 10.0,
        "fps": 30,
        "frame_size": "1920x1080",
        "summary": {"notes": "synthetic fixture"},
        "video_timeline": [
            {
                "clip": "clip_001.mp4",
                "start_s": 0.0,
                "end_s": 5.0,
                "duration_s": 5.0,
                "clip_occurrence": 0,
            }
        ],
        "audio_timeline": {
            "dedicated_audio_tracks": [
                {
                    "clip": "clip_001.mp4",
                    "start_s": 0.0,
                    "end_s": 5.0,
                    "duration_s": 5.0,
                    "track_label": "A1",
                }
            ]
        },
    }
    (project_dir / "timeline.json").write_text(json.dumps(timeline, indent=2), encoding="utf-8")
    # A dict (not a list) so save_output_to_prompts early-returns and leaves
    # video_prompts.json untouched when the fake workflow subprocess "finishes".
    (project_dir / "output.json").write_text(
        json.dumps({"smoke": True, "status": "ok"}), encoding="utf-8"
    )
    prompts = [
        {
            "prompt_version_id": "pv-smoke-0001",
            "matched_clip": "clip_001.mp4",
            "clip_used": "clip_001.mp4",
            "clip_occurrence": 0,
            "category": "video",
            "generation_type": "complex",
            "video_model_prompt": "A detailed cinematic smoke-test prompt for clip 1.",
            "selected_assets": [],
            "status": "completed",
            "timestamp": "2026-01-01T00:00:00Z",
            "history": [],
        }
    ]
    (project_dir / "video_prompts.json").write_text(json.dumps(prompts, indent=2), encoding="utf-8")


def _install_fakes():
    """Replace LLM calls and subprocesses with deterministic fakes.

    Every module that did `from <src module> import <name>` at its top holds
    its own binding to the original function, so a fake must replace the
    attribute in EVERY module that has it, not just the defining module.
    Sweeping sys.modules after `import server` covers server.py's top-level
    bindings, the src modules, and (once wired) the src.routers modules.
    """
    import server  # noqa: E402 -- env vars must be set before this import

    # --- LLM entry points ------------------------------------------------
    def fake_generate_structured(*args, **kwargs):
        return {"video_model_prompt": "Smoke-revised prompt for clip 1.", "explanation": "Smoke fake revision."}

    def fake_quality_check(*args, **kwargs):
        return {"score": 0.92, "summary": "fake quality report", "checks": [], "issues": []}

    def fake_learning_eval(*args, **kwargs):
        return {"score": 0.85, "eval_case_results": [], "summary": "fake learning report"}

    def fake_analyze_clip_context(*args, **kwargs):
        return {"summary": "Smoke fake clip context summary.", "categories": []}

    # --- chat reply ------------------------------------------------------
    async def fake_generate_chat_reply(*args, **kwargs):
        return ("Smoke fake assistant reply.", [])

    # --- import/export pipelines -----------------------------------------
    def fake_setup_project_workspace(*args, **kwargs):
        return ("/tmp/smoke_fake_proj", "/tmp/smoke_fake_proj/fake.prproj")

    def fake_extract_timeline_from_project(*args, **kwargs):
        return "/tmp/smoke_fake_proj/timeline.json"

    def fake_parse_and_align_feedback(*args, **kwargs):
        return "/tmp/smoke_fake_proj/feedback.json"

    def fake_extract_frame_at_offset(*args, **kwargs):
        return None

    # --- audio waveform --------------------------------------------------
    def fake_waveform(project_name, segment, bin_count):
        return {"bins": bin_count, "values": [0.1] * int(bin_count), "peak": 0.5}

    # --- workflow / video job execution ----------------------------------
    from src import job_manager

    def fake_execute_workflow_job(project, job_id):
        job_manager.update_project_job(project, job_id, {"status": "succeeded", "error": None})

    def fake_execute_video_generation_job(project, job_id):
        job_manager.update_project_job(project, job_id, {"status": "succeeded", "error": None})

    def fake_generate_clip_video(*args, **kwargs):
        return {"status": "queued", "job_id": "smoke-video-job", "message": "smoke fake"}

    # --- raw feedback mapping (run-workflow SSE) -------------------------
    def fake_ensure_raw_feedback(project):
        path = STORAGE_DIR / "data" / project / "raw_feedback_temp.json"
        if not path.exists():
            path.write_text(json.dumps({"schema_version": 1, "items": []}), encoding="utf-8")
        return str(path)

    fakes = {
        "generate_structured": fake_generate_structured,
        "run_quality_check": fake_quality_check,
        "run_learning_eval": fake_learning_eval,
        "analyze_clip_context": fake_analyze_clip_context,
        "generate_chat_reply_with_tools": fake_generate_chat_reply,
        "setup_project_workspace": fake_setup_project_workspace,
        "extract_timeline_from_project": fake_extract_timeline_from_project,
        "parse_and_align_feedback": fake_parse_and_align_feedback,
        "extract_frame_at_offset": fake_extract_frame_at_offset,
        "waveform_for_audio_segment": fake_waveform,
        "execute_workflow_job": fake_execute_workflow_job,
        "execute_video_generation_job": fake_execute_video_generation_job,
        "generate_clip_video": fake_generate_clip_video,
        "ensure_raw_feedback": fake_ensure_raw_feedback,
    }
    for module in list(sys.modules.values()):
        for name, fake in fakes.items():
            if hasattr(module, name):
                try:
                    setattr(module, name, fake)
                except (AttributeError, TypeError):
                    pass

    # --- subprocess execution (run-workflow SSE) -------------------------
    class FakeStream:
        def __init__(self, lines):
            self._lines = list(lines)

        async def readline(self):
            return self._lines.pop(0) if self._lines else b""

    class FakeProcess:
        def __init__(self):
            self.stdout = FakeStream([b"data: smoke fake subprocess line\n"])
            self.stderr = FakeStream([])

        async def wait(self):
            return 0

    async def fake_create_subprocess_exec(*args, **kwargs):
        return FakeProcess()

    asyncio.create_subprocess_exec = fake_create_subprocess_exec

    return server


def _resolve(value, state):
    """Recursively format {placeholder} strings using captured state."""
    if isinstance(value, dict):
        return {k: _resolve(v, state) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve(v, state) for v in value]
    if isinstance(value, str) and "{" in value:
        try:
            return value.format(**state)
        except (KeyError, IndexError):
            return value
    return value


def _path(path_template, state):
    try:
        return path_template.format(**state)
    except (KeyError, IndexError):
        return path_template


# label, method, path template, optional json/data/files, optional captures
# {p} = fixture project; captured ids: lesson_id, fb_id, case_id, job_id
ROUTES = [
    {"label": "GET /health", "method": "GET", "path": "/health"},
    {"label": "GET /version", "method": "GET", "path": "/version"},
    {"label": "GET /api/config/runtime", "method": "GET", "path": "/api/config/runtime"},
    {"label": "POST /api/config/secrets", "method": "POST", "path": "/api/config/secrets",
     "json": {"gemini_api_key": "smoke-gemini", "segmind_api_key": "smoke-segmind"}},
    {"label": "POST /api/projects", "method": "POST", "path": "/api/projects",
     "data": {"project_name": CREATED_PROJECT}},
    {"label": "GET /api/projects", "method": "GET", "path": "/api/projects"},
    {"label": "GET /api/project/{p}", "method": "GET", "path": "/api/project/{p}"},
    {"label": "GET /api/projects/{p}/clips/0/filmstrip", "method": "GET", "path": "/api/projects/{p}/clips/0/filmstrip"},
    {"label": "GET /api/projects/{p}/audio-waveform", "method": "GET", "path": "/api/projects/{p}/audio-waveform"},
    {"label": "GET /api/projects/{p}/clips/0/state", "method": "GET", "path": "/api/projects/{p}/clips/0/state",
     "capture": [("clip_key", ["clip_key"]), ("version_id", ["active_prompt", "version_id"])]},
    {"label": "GET /api/projects/{p}/clips/0/selection", "method": "GET", "path": "/api/projects/{p}/clips/0/selection"},
    {"label": "PATCH /api/projects/{p}/clips/0/selection", "method": "PATCH", "path": "/api/projects/{p}/clips/0/selection",
     "json": {"active_generated_video_id": "smoke-video-1"}},
    {"label": "POST /api/projects/{p}/assets/clips", "method": "POST", "path": "/api/projects/{p}/assets/clips",
     "files": [("files", ("placeholder.txt", b"smoke", "text/plain"))]},
    {"label": "POST /api/projects/{p}/premiere-package", "method": "POST", "path": "/api/projects/{p}/premiere-package",
     "files": {"package": ("pkg.zip", b"smoke", "application/zip")}},
    {"label": "POST /api/projects/{p}/feedback", "method": "POST", "path": "/api/projects/{p}/feedback",
     "files": {"file": ("feedback.txt", b"smoke feedback", "text/plain")}},
    {"label": "POST /api/projects/{p}/feedback/item", "method": "POST", "path": "/api/projects/{p}/feedback/item",
     "json": {"clip_used": "clip_001.mp4", "category": "timing", "remark": "too slow", "timestamp": "00:00:02:00"}},
    {"label": "GET /api/projects/{p}/events", "method": "GET", "path": "/api/projects/{p}/events"},
    {"label": "GET /api/projects/{p}/prompt-feedback", "method": "GET", "path": "/api/projects/{p}/prompt-feedback"},
    {"label": "GET /api/projects/{p}/prompt-lessons", "method": "GET", "path": "/api/projects/{p}/prompt-lessons"},
    {"label": "GET /api/projects/{p}/prompt-eval-cases", "method": "GET", "path": "/api/projects/{p}/prompt-eval-cases"},
    {"label": "POST /api/projects/{p}/prompt-lessons", "method": "POST", "path": "/api/projects/{p}/prompt-lessons",
     "json": {"scope": "clip", "clip_key": "{clip_key}", "category": "composition",
              "lesson": "Keep the subject centered.", "confidence": 0.9},
     "capture": [("lesson_id", ["lesson", "id"])]},
    {"label": "PATCH /api/projects/{p}/prompt-lessons/{lesson_id}", "method": "PATCH", "path": "/api/projects/{p}/prompt-lessons/{lesson_id}",
     "json": {"archived": True}},
    {"label": "POST /api/projects/{p}/prompt-feedback", "method": "POST", "path": "/api/projects/{p}/prompt-feedback",
     "json": {"clip_index": 0, "clip_key": "{clip_key}", "prompt_id": "smoke-prompt",
              "prompt_version_id": "{version_id}", "rating": "negative",
              "categories": ["wrong_visual_detail"], "severity": 4, "comment": "Subject drifts off center."},
     "capture": [("fb_id", ["item", "id"])]},
    {"label": "PATCH /api/projects/{p}/prompt-feedback/{fb_id}", "method": "PATCH", "path": "/api/projects/{p}/prompt-feedback/{fb_id}",
     "json": {"status": "open"}},
    {"label": "POST /api/projects/{p}/prompt-feedback/{fb_id}/eval-case", "method": "POST", "path": "/api/projects/{p}/prompt-feedback/{fb_id}/eval-case",
     "capture": [("case_id", ["eval_case", "id"])]},
    {"label": "PATCH /api/projects/{p}/prompt-eval-cases/{case_id}", "method": "PATCH", "path": "/api/projects/{p}/prompt-eval-cases/{case_id}",
     "json": {"enabled": True}},
    {"label": "POST /api/projects/{p}/prompt-feedback/{fb_id}/suggest-lesson", "method": "POST", "path": "/api/projects/{p}/prompt-feedback/{fb_id}/suggest-lesson",
     "json": {"provider": "openai"}},
    {"label": "POST /api/projects/{p}/prompts/{version_id}/revise-from-feedback", "method": "POST", "path": "/api/projects/{p}/prompts/{version_id}/revise-from-feedback",
     "json": {"clip_index": 0, "feedback_ids": ["{fb_id}"], "provider": "openai"}},
    {"label": "GET /api/projects/{p}/chat/clip/0", "method": "GET", "path": "/api/projects/{p}/chat/clip/0"},
    {"label": "POST /api/projects/{p}/chat/clip", "method": "POST", "path": "/api/projects/{p}/chat/clip",
     "json": {"clip_index": 0, "message": "hello", "provider": "openai"}},
    {"label": "POST /api/projects/{p}/chat/clip/action", "method": "POST", "path": "/api/projects/{p}/chat/clip/action",
     "json": {"clip_index": 0, "message": "save this feedback", "provider": "openai",
              "action": {"type": "save_prompt_feedback", "comment": "lighting too dark",
                         "correction": "brighten the face", "prompt_version_id": "pv-smoke-0001"}}},
    {"label": "POST /api/projects/{p}/chat/memory", "method": "POST", "path": "/api/projects/{p}/chat/memory",
     "json": {"clip_index": 0, "text": "Remember: the hero car is blue.", "scope": "clip"}},
    {"label": "POST /api/projects/{p}/generate-video", "method": "POST", "path": "/api/projects/{p}/generate-video",
     "json": {"clip_index": 0, "duration": 5, "resolution": "720p", "aspect_ratio": "9:16"}},
    {"label": "POST /api/projects/{p}/jobs/workflow", "method": "POST", "path": "/api/projects/{p}/jobs/workflow",
     "json": {"feedback_index": 0, "provider": "openai"},
     "capture": [("job_id", ["id"])]},
    {"label": "POST /api/projects/{p}/jobs/generate-video", "method": "POST", "path": "/api/projects/{p}/jobs/generate-video",
     "json": {"clip_index": 0, "duration": 5}},
    {"label": "GET /api/projects/{p}/jobs", "method": "GET", "path": "/api/projects/{p}/jobs"},
    {"label": "GET /api/projects/{p}/jobs/{job_id}", "method": "GET", "path": "/api/projects/{p}/jobs/{job_id}"},
    {"label": "POST /api/projects/{p}/jobs/{job_id}/cancel", "method": "POST", "path": "/api/projects/{p}/jobs/{job_id}/cancel"},
    {"label": "GET /api/run-workflow", "method": "GET", "path": "/api/run-workflow?project={p}&index=0"},
    {"label": "GET /api/workflow-result", "method": "GET", "path": "/api/workflow-result?project={p}"},
]


def run():
    build_fixture()
    os.environ["LOKA_STORAGE_DIR"] = str(STORAGE_DIR)
    os.environ["OPENAI_API_KEY"] = "smoke-fake-openai-key"
    server = _install_fakes()
    from fastapi.testclient import TestClient

    client = TestClient(server.app, raise_server_exceptions=False)
    state = {"p": FIXTURE_PROJECT}
    results = {}

    for spec in ROUTES:
        label = spec["label"]
        kwargs = {}
        if spec.get("json"):
            kwargs["json"] = _resolve(spec["json"], state)
        if spec.get("data"):
            kwargs["data"] = _resolve(spec["data"], state)
        if spec.get("files"):
            kwargs["files"] = spec["files"]
        try:
            response = client.request(spec["method"], _path(spec["path"], state), **kwargs)
            content_type = response.headers.get("content-type", "")
            if "application/json" in content_type:
                try:
                    raw_body = response.json()
                except Exception:
                    raw_body = response.text
            else:
                raw_body = response.text
            result = {"status": response.status_code, "body": normalize(raw_body)}
        except Exception as exc:  # noqa: BLE001 - record any failure, compare consistently
            raw_body = None
            result = {"status": "EXC", "body": f"{type(exc).__name__}: {exc}"}
        results[label] = result

        # Captures read the RAW body: ids must survive verbatim for later paths.
        for key, path_parts in spec.get("capture", []):
            if (
                isinstance(result["status"], int)
                and 200 <= result["status"] < 300
                and isinstance(raw_body, dict)
            ):
                node = raw_body
                try:
                    for part in path_parts:
                        node = node[part]
                    state[key] = node
                except (KeyError, TypeError):
                    pass

    return results


def _print_summary(results):
    counts = {}
    for result in results.values():
        key = str(result["status"])
        counts[key] = counts.get(key, 0) + 1
    summary = ", ".join(f"{k}: {v}" for k, v in sorted(counts.items()))
    print(f"Routes executed: {len(results)}  ({summary})")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "compare"
    results = run()
    _print_summary(results)

    if mode == "baseline":
        BASELINE_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Baseline recorded: {BASELINE_PATH}")
        return 0

    if not BASELINE_PATH.exists():
        print("No baseline found. Run: python scripts/smoke_refactor.py baseline", file=sys.stderr)
        return 2

    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    diffs = []
    for label, current in results.items():
        expected = baseline.get(label)
        if expected is None:
            diffs.append((label, "MISSING IN BASELINE", current))
        elif expected != current:
            diffs.append((label, expected, current))

    if diffs:
        print(f"\n{len(diffs)} route(s) DIFFER from baseline:")
        for label, expected, current in diffs:
            print(f"\n  {label}")
            print(f"    baseline: {json.dumps(expected, ensure_ascii=False)[:400]}")
            print(f"    current : {json.dumps(current, ensure_ascii=False)[:400]}")
        return 1

    print("\nAll routes match baseline. No wiring regressions.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
