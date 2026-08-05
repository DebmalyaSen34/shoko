# Shoko Backend — API Reference

FastAPI server for the **Loka15 Studio** desktop application (branded "Shoko"). Ingests Adobe Premiere Pro project timelines and client feedback, generates AI video prompts via OpenAI/Gemini, and runs Seedance video generation jobs.

Served from `server.py`. For the full project overview including the React frontend and Tauri desktop shell, see the [root README](../README.md).

## Table of Contents

- [Architecture](#architecture)
- [Requirements](#requirements)
- [Setup & Running](#setup--running)
- [Runtime Configuration](#runtime-configuration)
- [Project Data Layout](#project-data-layout)
- [Conventions](#conventions)
- [API Quick Reference](#api-quick-reference)
- [API Endpoints](#api-endpoints)
  - [System](#system)
  - [Projects](#projects)
  - [Clips](#clips)
  - [Feedback & Prompt Learning](#feedback--prompt-learning)
  - [Jobs & Workflow](#jobs--workflow)
  - [Clip Chat](#clip-chat)
- [Server-Sent Events (SSE)](#server-sent-events-sse)
- [Static File Mounts](#static-file-mounts)
- [OpenAPI Docs](#openapi-docs)
- [Testing](#testing)
- [Security & Operational Caveats](#security--operational-caveats)
- [Known Inconsistencies](#known-inconsistencies)
- [Related Docs](#related-docs)

## Architecture

```
server.py                    FastAPI app entry point
├── CORSMiddleware            allow_origins=["*"]
├── HTTP request logger       logs every request + errors to data/logs/backend.jsonl
├── Routers (6)               no URL prefix, no auth dependencies
│   ├── system_router         /health, /version, /shutdown, /api/config/*
│   ├── projects_router       /api/projects/*, /api/project/*
│   ├── clips_router          /api/projects/{p}/clips/*
│   ├── feedback_router       /api/projects/{p}/prompt-*
│   ├── jobs_router           /api/projects/{p}/jobs/*, /api/run-workflow
│   └── chat_router           /api/projects/{p}/chat/*
├── Static mounts             /assets → ASSETS_DIR, /data → DATA_DIR
└── uvicorn                   port: LOKA_BACKEND_PORT → PORT → 8000
```

Domain modules live under `src/`: parsers, generators, workflows, storage helpers, and the clip state builder. FastAPI auto-generates OpenAPI docs at `/docs` (Swagger UI) and `/redoc` while the server is running.

## Requirements

- **Python 3.10+**
- **ffmpeg** and **ffprobe** on `PATH` (media extraction, waveform generation)
- Provider API keys (at least one, depending on usage):
  - `OPENAI_API_KEY`
  - `GEMINI_API_KEY`
  - `SEGMIND_API_KEY`
- Python dependencies: `pip install -r requirements.txt`

## Setup & Running

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and fill in your provider API keys
```

Start the server:

```bash
python -m uvicorn server:app --host 127.0.0.1 --port 8000 --reload
# or:
python backend_server.py
```

Port resolution: `LOKA_BACKEND_PORT` → `PORT` → `8000`.

## Runtime Configuration

The backend loads configuration in this precedence order:

1. Real OS environment variables
2. `LOKA_ENV_FILE` (when set)
3. Installed-app config at `{app storage}/config/.env`
4. Local development file at `backend/.env`
5. Code defaults

### Environment Variables

| Variable | Description | Default |
|---|---|---|
| `OPENAI_API_KEY` | OpenAI API key | — |
| `GEMINI_API_KEY` | Google Gemini API key | — |
| `SEGMIND_API_KEY` | Segmind API key (Seedance video) | — |
| `LOKA_BACKEND_PORT` | Server listen port | `8000` |
| `LOKA_STORAGE_DIR` | Override storage root | OS-specific (see below) |
| `LOKA_ENV_FILE` | Path to additional `.env` file | — |
| `LOKA_BACKEND_SHUTDOWN_TOKEN` | Token required for `/shutdown` | — |
| `LOKA_APP_VERSION` | Override reported version | from `version.txt` |
| `LOKA_ASSET_CACHE_BACKEND` | Asset URL cache backend (`sqlite` or `supabase`) | `sqlite` |
| `SUPABASE_URL` | Supabase project URL (when using Supabase cache) | — |
| `SUPABASE_SERVICE_ROLE_KEY` | Supabase service role key | — |
| `LOKA_ENABLE_REFERENCE_AUDIO_UPLOAD` | Enable audio upload feature flag | — |

Default app storage paths:

| OS | Path |
|---|---|
| macOS | `~/Library/Application Support/Loka15 Studio` |
| Windows | `%LOCALAPPDATA%\Loka15 Studio` |
| Linux | `${XDG_DATA_HOME:-~/.local/share}/Loka15 Studio` |

Set `LOKA_STORAGE_DIR` as a real OS environment variable to override the storage root.

## Project Data Layout

```
{storage}/
├── data/{project}/
│   ├── timeline.json           extracted Premiere timeline
│   ├── feedback.json           parsed & aligned feedback
│   ├── video_prompts.json      generated prompt versions
│   ├── prompt_feedback.json    user feedback on prompts
│   ├── prompt_lessons.json     learned prompt improvements
│   ├── prompt_eval_cases.json  evaluation test cases
│   ├── jobs.json               async job records
│   ├── clip_selections.json    per-clip user selections
│   ├── events.jsonl            append-only project event log
│   ├── clip_chats.json         chat history per clip
│   ├── chat_memory.json        persistent chat memory items
│   └── chat_agent_runs.json    agent run records
└── assets/{project}/
    ├── 00_style/
    ├── 01_characters/
    ├── 02_props/
    ├── 03_locations/
    ├── 04_audio/
    ├── 05_references/
    └── 06_clips/
        ├── _final/
        └── _raw/
```

Data is stored as JSON files per project — there is no database ORM. An optional SQLite database at `{storage}/data/asset_url_cache.sqlite3` caches asset URLs (default; Supabase alternative available via `LOKA_ASSET_CACHE_BACKEND`).

## Conventions

- **Base URL:** `http://127.0.0.1:8000`
- **Project names** are sanitized via `safe_project_name()` — path separators and special characters are rejected or stripped
- **Clip indices** are 0-based (position in `video_timeline`)
- **Error responses** use FastAPI's default `{"detail": "..."}` format
- **Timestamps** are ISO 8601 UTC strings
- **PATCH endpoints** use `exclude_none=True` — only the fields you include are updated; omitted fields are left unchanged
- **Responses** are raw Python dicts serialized by FastAPI — no `response_model` Pydantic validation on the way out

## API Quick Reference

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/version` | Server version info |
| `POST` | `/shutdown` | Shut down the server (token-gated) |
| `GET` | `/api/config/runtime` | Get runtime config & key status |
| `POST` | `/api/config/secrets` | Update API keys in config file |
| `GET` | `/api/projects` | List all projects |
| `POST` | `/api/projects` | Create a new project |
| `POST` | `/api/projects/{p}/assets/{category}` | Upload asset files |
| `POST` | `/api/projects/{p}/premiere-package` | Import Premiere Pro zip |
| `POST` | `/api/projects/{p}/feedback` | Upload & parse feedback file |
| `POST` | `/api/projects/{p}/feedback/item` | Add manual feedback item |
| `GET` | `/api/project/{p}` | Get full project data & timeline |
| `GET` | `/api/projects/{p}/audio-waveform` | Get audio waveform data |
| `GET` | `/api/projects/{p}/events` | Get project event log |
| `GET` | `/api/projects/{p}/clips/{i}/filmstrip` | Get clip filmstrip frames |
| `GET` | `/api/projects/{p}/clips/{i}/state` | Get full clip state |
| `GET` | `/api/projects/{p}/clips/{i}/selection` | Get clip user selections |
| `PATCH` | `/api/projects/{p}/clips/{i}/selection` | Update clip user selections |
| `GET` | `/api/projects/{p}/prompt-feedback` | List prompt feedback items |
| `POST` | `/api/projects/{p}/prompt-feedback` | Create prompt feedback item |
| `PATCH` | `/api/projects/{p}/prompt-feedback/{id}` | Update prompt feedback item |
| `POST` | `/api/projects/{p}/prompt-feedback/{id}/eval-case` | Create eval case from feedback |
| `POST` | `/api/projects/{p}/prompt-feedback/{id}/suggest-lesson` | AI-suggest a lesson from feedback |
| `GET` | `/api/projects/{p}/prompt-lessons` | List prompt lessons |
| `POST` | `/api/projects/{p}/prompt-lessons` | Create a prompt lesson |
| `PATCH` | `/api/projects/{p}/prompt-lessons/{id}` | Update a prompt lesson |
| `GET` | `/api/projects/{p}/prompt-eval-cases` | List eval cases |
| `PATCH` | `/api/projects/{p}/prompt-eval-cases/{id}` | Update an eval case |
| `POST` | `/api/projects/{p}/prompts/{vid}/revise-from-feedback` | Revise a prompt using feedback |
| `POST` | `/api/projects/{p}/generate-video` | Generate video (sync/blocking) |
| `GET` | `/api/projects/{p}/jobs` | List recent project jobs |
| `GET` | `/api/projects/{p}/jobs/{job_id}` | Get a specific job |
| `POST` | `/api/projects/{p}/jobs/{job_id}/cancel` | Cancel a running job |
| `POST` | `/api/projects/{p}/jobs/workflow` | Create & enqueue a workflow job |
| `POST` | `/api/projects/{p}/jobs/generate-video` | Create & enqueue a video job |
| `GET` | `/api/run-workflow` | Run workflow with SSE streaming |
| `GET` | `/api/workflow-result` | Get last workflow output |
| `GET` | `/api/projects/{p}/chat/clip/{i}` | Get clip chat history |
| `POST` | `/api/projects/{p}/chat/clip` | Send a chat message |
| `POST` | `/api/projects/{p}/chat/clip/action` | Execute a chat action |
| `POST` | `/api/projects/{p}/chat/memory` | Save a chat memory item |

## API Endpoints

### System

Health, version, shutdown, and runtime configuration.

---

#### `GET /health`

Health check. Returns server status and uptime.

**Response 200:**
```json
{
  "status": "ok",
  "service": "loka15-backend",
  "version": "0.7.1",
  "started_at": "2026-08-04T10:30:00.123456"
}
```

```bash
curl http://127.0.0.1:8000/health
```

---

#### `GET /version`

Server version and Python version.

**Response 200:**
```json
{
  "service": "loka15-backend",
  "version": "0.7.1",
  "python": "3.12.3"
}
```

```bash
curl http://127.0.0.1:8000/version
```

---

#### `POST /shutdown`

Shut down the server process. Requires a matching token header.

> ⚠️ This is the **only** authenticated endpoint.

**Headers:**
| Header | Required | Description |
|---|---|---|
| `x-loka-shutdown-token` | yes | Must match `LOKA_BACKEND_SHUTDOWN_TOKEN` env var |

**Response 200:**
```json
{"status": "shutting_down"}
```

**Errors:** 403 (missing or invalid token)

```bash
curl -X POST http://127.0.0.1:8000/shutdown \
  -H "x-loka-shutdown-token: my-secret-token"
```

---

#### `GET /api/config/runtime`

Get runtime configuration including provider key status (keys are never returned, only whether they are set).

**Response 200:**
```json
{
  "openai_api_key_set": true,
  "gemini_api_key_set": true,
  "segmind_api_key_set": false,
  "storage_dir": "/Users/.../Library/Application Support/Loka15 Studio",
  "data_dir": "/Users/.../Library/Application Support/Loka15 Studio/data",
  "assets_dir": "/Users/.../Library/Application Support/Loka15 Studio/assets"
}
```

```bash
curl http://127.0.0.1:8000/api/config/runtime
```

---

#### `POST /api/config/secrets`

Update API keys in the config file. Only keys not locked by real OS environment variables can be changed.

> ⚠️ This endpoint is **unauthenticated**. It writes secrets to a chmod 600 config file on disk.

**Body (JSON):**
```json
{
  "openai_api_key": "sk-...",
  "gemini_api_key": "ai-...",
  "segmind_api_key": "sg-..."
}
```

All fields are optional. Pass an empty string `""` to clear a key.

| Field | Type | Required | Description |
|---|---|---|---|
| `openai_api_key` | string | no | OpenAI API key |
| `gemini_api_key` | string | no | Google Gemini API key |
| `segmind_api_key` | string | no | Segmind API key |

**Response 200:**
```json
{
  "openai_api_key_set": true,
  "gemini_api_key_set": true,
  "segmind_api_key_set": true,
  "locked_keys": []
}
```

**Errors:** 409 (one or more keys are locked by real OS env vars)

```bash
curl -X POST http://127.0.0.1:8000/api/config/secrets \
  -H "Content-Type: application/json" \
  -d '{"openai_api_key": "sk-..."}'
```

### Projects

Project CRUD, asset uploads, Premiere package import, feedback upload, and project data queries.

---

#### `GET /api/projects`

List all projects that have a `timeline.json`.

**Response 200:**
```json
["demo", "short-film", "test-project"]
```

```bash
curl http://127.0.0.1:8000/api/projects
```

---

#### `POST /api/projects`

Create a new project workspace with the standard asset directory tree.

**Body:** `multipart/form-data`
| Field | Type | Required | Description |
|---|---|---|---|
| `project_name` | string | yes | Name for the new project |

**Response 200:**
```json
{
  "project_name": "my-project",
  "assets_dir": "/Users/.../assets/my-project",
  "required_directories": ["00_style", "01_characters", "02_props", "03_locations", "04_audio", "05_references", "06_clips"]
}
```

**Errors:** 409 (project already exists)

```bash
curl -X POST http://127.0.0.1:8000/api/projects \
  -F "project_name=my-project"
```

---

#### `POST /api/projects/{project_name}/assets/{category}`

Upload one or more asset files into a project category directory.

**Path params:**
| Param | Description |
|---|---|
| `project_name` | Project name |
| `category` | Asset category (see table below) |

**Asset categories:**
| Category | Directory |
|---|---|
| `style` | `00_style/` |
| `characters` | `01_characters/` |
| `props` | `02_props/` |
| `locations` | `03_locations/` |
| `audio` | `04_audio/` |
| `references` | `05_references/` |
| `clips` | `06_clips/` |

**Body:** `multipart/form-data` — one or more files in the `files` field.

**Response 200:**
```json
{
  "project_name": "demo",
  "category": "references",
  "files": ["05_references/mood-board.jpg", "05_references/style-guide.png"]
}
```

**Errors:** 400 (invalid category)

```bash
curl -X POST http://127.0.0.1:8000/api/projects/demo/assets/references \
  -F "files=@mood-board.jpg" \
  -F "files=@style-guide.png"
```

---

#### `POST /api/projects/{project_name}/premiere-package`

Upload a `.zip` Premiere Pro package. Extracts the `.prproj` file, parses the timeline, copies source media into the project asset tree, and creates `timeline.json`.

**Body:** `multipart/form-data`
| Field | Type | Required | Description |
|---|---|---|---|
| `package` | File | yes | `.zip` file containing a Premiere project |

**Response 200:**
```json
{
  "project_name": "demo",
  "timeline_path": "/Users/.../data/demo/timeline.json"
}
```

**Errors:** 400 (not a `.zip` file, invalid package), 500 (extraction/parsing failure)

```bash
curl -X POST http://127.0.0.1:8000/api/projects/demo/premiere-package \
  -F "package=@my-project-export.zip"
```

---

#### `POST /api/projects/{project_name}/feedback`

Upload a feedback file (`.txt`, `.csv`, `.xlsx`, or `.xls`). The server parses it and aligns remarks to timeline clips using AI.

> Requires `OPENAI_API_KEY` to be set. The project must already have a `timeline.json` (import a Premiere package first).

**Body:** `multipart/form-data`
| Field | Type | Required | Description |
|---|---|---|---|
| `file` | File | yes | `.txt`, `.csv`, `.xlsx`, or `.xls` |

**Response 200:**
```json
{
  "project_name": "demo",
  "feedback_path": "/Users/.../data/demo/feedback.json"
}
```

**Errors:** 400 (no timeline yet, unsupported file extension), 500 (`OPENAI_API_KEY` not set, parsing failure)

```bash
curl -X POST http://127.0.0.1:8000/api/projects/demo/feedback \
  -F "file=@client_notes.csv"
```

---

#### `POST /api/projects/{project_name}/feedback/item`

Add a single manual feedback remark without uploading a file.

**Body (JSON):**
```json
{
  "clip_used": "Clip_01.mp4",
  "category": "video",
  "remark": "The transition at 0:12 is too abrupt.",
  "timestamp": "00:00:12"
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `clip_used` | string | yes | Clip name the remark applies to |
| `category` | string | yes | Category (`audio`, `video`, `both`, etc.) |
| `remark` | string | yes | The feedback text |
| `timestamp` | string | no | Timecode for the remark |

**Response 200:**
```json
{"status": "success", "project_name": "demo"}
```

```bash
curl -X POST http://127.0.0.1:8000/api/projects/demo/feedback/item \
  -H "Content-Type: application/json" \
  -d '{"clip_used": "Clip_01.mp4", "category": "video", "remark": "The transition at 0:12 is too abrupt."}'
```

---

#### `GET /api/project/{project_name}`

Get the full project data: timeline, assets, feedback, prompt history, and all derived artifacts.

> ⚠️ **URL inconsistency:** This endpoint uses the singular `project` (not `projects` like all other project-scoped routes).

**Response 200:** A large nested object containing:
```json
{
  "project_name": "demo",
  "schema_version": 2,
  "timeline": { "video_timeline": [...], "audio_timeline": {...}, "total_duration_s": 120.5 },
  "assets": { "style": [...], "characters": [...], "clips": [...] },
  "feedback": { "feedback_groups": [...] },
  "active_prompt": { "versions": [...] },
  "..."
}
```

```bash
curl http://127.0.0.1:8000/api/project/demo
```

---

#### `GET /api/projects/{project_name}/audio-waveform`

Get audio waveform amplitude data for visualization. Supports master track only or all tracks.

**Query params:**
| Param | Type | Default | Description |
|---|---|---|---|
| `bins` | int | `600` | Number of amplitude samples (120–1800) |
| `mode` | string | `"master"` | `"master"` for master audio only, `"all"` for all tracks |

**Response 200:**
```json
{
  "project_name": "demo",
  "bins": 600,
  "mode": "master",
  "segments": [
    {
      "audio_index": 0,
      "start_s": 0.0,
      "end_s": 30.5,
      "duration_s": 30.5,
      "waveform": [0.12, 0.45, 0.78, "..."]
    }
  ]
}
```

**Errors:** 404 (no timeline)

```bash
curl "http://127.0.0.1:8000/api/projects/demo/audio-waveform?bins=300&mode=all"
```

---

#### `GET /api/projects/{project_name}/events`

Get the project's append-only event log, filtered by optional criteria.

**Query params:**
| Param | Type | Default | Description |
|---|---|---|---|
| `limit` | int | `200` | Max events to return |
| `clip_key` | string | — | Filter by clip key |
| `event_type` | string | — | Filter by event type (e.g. `prompt_version_revised_from_feedback`) |

**Response 200:**
```json
{
  "project_name": "demo",
  "events": [
    {"timestamp": "2026-08-04T10:30:00Z", "event_type": "project_created", "actor": "user", "..."},
    {"timestamp": "2026-08-04T10:31:00Z", "event_type": "premiere_package_imported", "actor": "user", "..."}
  ]
}
```

```bash
curl "http://127.0.0.1:8000/api/projects/demo/events?limit=50&event_type=prompt_version_revised_from_feedback"
```

### Clips

Filmstrip generation, clip state, and user selection management.

---

#### `GET /api/projects/{project_name}/clips/{clip_index}/filmstrip`

Generate (or retrieve cached) filmstrip frame images for a clip. Extracts evenly-spaced frames from the source video.

**Query params:**
| Param | Type | Default | Description |
|---|---|---|---|
| `frames` | int | `4` | Number of frames to extract (1–8) |

**Response 200:**
```json
{
  "clip_index": 2,
  "frames": [
    {"offset_s": 0.0, "url": "/data/demo/timeline_filmstrips/0002_Clip_02/frame_01.jpg"},
    {"offset_s": 3.5, "url": "/data/demo/timeline_filmstrips/0002_Clip_02/frame_02.jpg"},
    {"offset_s": 7.0, "url": "/data/demo/timeline_filmstrips/0002_Clip_02/frame_03.jpg"},
    {"offset_s": 10.5, "url": "/data/demo/timeline_filmstrips/0002_Clip_02/frame_04.jpg"}
  ]
}
```

Frame URLs are served from the `/data` static mount. Generated frames are cached on disk for subsequent requests.

**Errors:** 404 (no timeline, clip index out of range)

```bash
curl "http://127.0.0.1:8000/api/projects/demo/clips/2/filmstrip?frames=6"
```

---

#### `GET /api/projects/{project_name}/clips/{clip_index}/state`

Get the full compound state for a clip: timeline context, feedback, active prompt with versions, assets, selection, video state, analysis, memory, learning, agent runs, and jobs.

**Response 200:** Large nested object with top-level keys:
```json
{
  "schema_version": 2,
  "project_name": "demo",
  "clip_index": 2,
  "clip_key": "...",
  "clip_state_id": "...",
  "updated_at": "2026-08-04T10:30:00Z",
  "project": {...},
  "timeline": {"clip": {...}, "adjacent_clips": [...]},
  "feedback_state": {...},
  "active_prompt": {"version": {...}, "versions": [...]},
  "asset_state": {...},
  "video_state": {...},
  "selection_state": {...},
  "analysis_state": {...},
  "memory_state": {...},
  "learning_state": {...},
  "agent_state": {...},
  "job_state": {...},
  "freshness": {...}
}
```

**Errors:** 404 (no timeline, clip index out of range)

```bash
curl http://127.0.0.1:8000/api/projects/demo/clips/2/state
```

---

#### `GET /api/projects/{project_name}/clips/{clip_index}/selection`

Get the user's persisted selections for a clip — active prompt version, generated video, and selected/pinned assets.

**Response 200:**
```json
{
  "selection": {
    "active_prompt_version_id": "abc123...",
    "active_generated_video_id": null,
    "selected_assets": [...],
    "pinned_assets": [...],
    "selected_asset_ids": ["asset-1", "asset-2"],
    "pinned_asset_ids": []
  },
  "clip_state": { "..." }
}
```

```bash
curl http://127.0.0.1:8000/api/projects/demo/clips/2/selection
```

---

#### `PATCH /api/projects/{project_name}/clips/{clip_index}/selection`

Update persisted selections for a clip. Omitted fields are left unchanged.

**Body (JSON):** All fields optional (PATCH semantics):
```json
{
  "active_prompt_version_id": "abc123...",
  "active_generated_video_id": null,
  "selected_assets": [{"path": "05_references/img.jpg", "label": "Reference"}],
  "pinned_assets": [],
  "selected_asset_ids": ["asset-1"],
  "selected_asset_paths": ["05_references/img.jpg"],
  "pinned_asset_ids": [],
  "pinned_asset_paths": []
}
```

| Field | Type | Description |
|---|---|---|
| `active_prompt_version_id` | string | Set the active prompt version |
| `active_generated_video_id` | string | Set the active generated video |
| `selected_assets` | list[dict] | Asset objects with path/label |
| `pinned_assets` | list[dict] | Pinned asset objects |
| `selected_asset_ids` | list[string] | Selected asset IDs |
| `selected_asset_paths` | list[string] | Selected asset paths |
| `pinned_asset_ids` | list[string] | Pinned asset IDs |
| `pinned_asset_paths` | list[string] | Pinned asset paths |

**Response 200:**
```json
{
  "selection": {"active_prompt_version_id": "abc123...", "selected_asset_ids": ["asset-1"], "..."},
  "clip_state": { "..." }
}
```

```bash
curl -X PATCH http://127.0.0.1:8000/api/projects/demo/clips/2/selection \
  -H "Content-Type: application/json" \
  -d '{"active_prompt_version_id": "abc123..."}'
```

### Feedback & Prompt Learning

Prompt feedback collection, lesson management, evaluation cases, and AI-assisted prompt revision.

---

#### `GET /api/projects/{project_name}/prompt-feedback`

List prompt feedback items, optionally filtered by clip.

**Query params:**
| Param | Type | Description |
|---|---|---|
| `clip_index` | int | Filter by clip index (optional) |

**Response 200:**
```json
{
  "schema_version": 1,
  "project_name": "demo",
  "clip_index": 2,
  "items": [
    {
      "id": "fb-001",
      "clip_index": 2,
      "prompt_version_id": "abc123",
      "rating": "negative",
      "categories": ["continuity"],
      "severity": 3,
      "comment": "Character position inconsistent with previous clip.",
      "correction": "Keep character at left edge of frame.",
      "status": "open",
      "created_at": "2026-08-04T10:30:00Z"
    }
  ],
  "summaries": {"abc123": {"positive": 2, "negative": 1, "total": 3}}
}
```

```bash
curl "http://127.0.0.1:8000/api/projects/demo/prompt-feedback?clip_index=2"
```

---

#### `POST /api/projects/{project_name}/prompt-feedback`

Create a new prompt feedback item.

**Body (JSON):**
```json
{
  "clip_index": 2,
  "clip_key": "clip_002",
  "prompt_id": "prompt-001",
  "prompt_version_id": "abc123",
  "rating": "negative",
  "categories": ["continuity"],
  "severity": 3,
  "comment": "Character position inconsistent.",
  "correction": "Keep character at left edge.",
  "remember_note": "",
  "create_eval_case": false,
  "status": "open"
}
```

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `clip_index` | int | yes | — | Clip index (0-based) |
| `clip_key` | string | yes | — | Clip key identifier |
| `prompt_id` | string | yes | — | Parent prompt ID |
| `prompt_version_id` | string | yes | — | Prompt version this feedback targets |
| `rating` | string | yes | — | `"positive"` or `"negative"` |
| `categories` | list[string] | no | `[]` | Feedback categories |
| `severity` | int | no | `3` | Severity level |
| `comment` | string | no | `""` | Feedback comment |
| `correction` | string | no | `""` | Suggested correction |
| `remember_note` | string | no | `""` | Note to remember for future prompts |
| `create_eval_case` | bool | no | `false` | Auto-create an eval case |
| `status` | string | no | — | `"open"`, `"approved"`, `"rejected"`, or `"resolved"` |

**Response 200:**
```json
{
  "item": {"id": "fb-001", "clip_index": 2, "rating": "negative", "..."},
  "clip_state": { "..." }
}
```

If `create_eval_case` is true, the response also includes an `eval_case` key.

```bash
curl -X POST http://127.0.0.1:8000/api/projects/demo/prompt-feedback \
  -H "Content-Type: application/json" \
  -d '{"clip_index": 2, "clip_key": "clip_002", "prompt_id": "prompt-001", "prompt_version_id": "abc123", "rating": "negative", "comment": "Bad continuity."}'
```

---

#### `PATCH /api/projects/{project_name}/prompt-feedback/{feedback_id}`

Update an existing prompt feedback item. Omitted fields are left unchanged.

**Body (JSON):** All optional:
```json
{
  "rating": "positive",
  "status": "resolved",
  "comment": "Fixed in revision."
}
```

Updatable fields: `rating`, `categories`, `severity`, `comment`, `correction`, `remember_note`, `create_eval_case`, `status`.

**Response 200:**
```json
{
  "item": {"id": "fb-001", "rating": "positive", "status": "resolved", "..."},
  "clip_state": { "..." }
}
```

**Errors:** 404 (feedback not found)

```bash
curl -X PATCH http://127.0.0.1:8000/api/projects/demo/prompt-feedback/fb-001 \
  -H "Content-Type: application/json" \
  -d '{"status": "resolved"}'
```

---

#### `POST /api/projects/{project_name}/prompt-feedback/{feedback_id}/eval-case`

Create a prompt evaluation test case from a negative feedback item.

> Only works for feedback items with `rating: "negative"`.

**Response 200:**
```json
{
  "eval_case": {
    "id": "ec-001",
    "name": "Continuity check",
    "input": {...},
    "expected_behavior": ["Character stays at left edge"],
    "failure_categories": ["continuity"],
    "enabled": true,
    "clip_index": 2
  },
  "clip_state": { "..." }
}
```

**Errors:** 404 (feedback not found), 400 (not a negative rating)

```bash
curl -X POST http://127.0.0.1:8000/api/projects/demo/prompt-feedback/fb-001/eval-case
```

---

#### `POST /api/projects/{project_name}/prompt-feedback/{feedback_id}/suggest-lesson`

Use AI to generate a prompt lesson suggestion from a feedback item.

**Body (JSON):**
| Field | Type | Default | Description |
|---|---|---|---|
| `provider` | string | `"openai"` | AI provider (`"openai"` or `"gemini"`) |

**Response 200:**
```json
{
  "project_name": "demo",
  "feedback_id": "fb-001",
  "suggestion": {
    "lesson": "When a clip follows a character close-up, maintain the character's screen position from the previous clip.",
    "category": "continuity",
    "confidence": 0.88,
    "reasoning": "The feedback mentions inconsistent character positioning across clips."
  }
}
```

**Errors:** 404 (feedback not found)

```bash
curl -X POST http://127.0.0.1:8000/api/projects/demo/prompt-feedback/fb-001/suggest-lesson \
  -H "Content-Type: application/json" \
  -d '{"provider": "openai"}'
```

---

#### `GET /api/projects/{project_name}/prompt-lessons`

List prompt learning lessons, with optional filters.

**Query params:**
| Param | Type | Default | Description |
|---|---|---|---|
| `clip_key` | string | — | Filter by clip key |
| `category` | string | — | Filter by lesson category |
| `include_archived` | bool | `false` | Include archived lessons |

**Response 200:**
```json
{
  "schema_version": 1,
  "project_name": "demo",
  "clip_key": null,
  "category": null,
  "lessons": [
    {
      "id": "lesson-001",
      "scope": "project",
      "category": "continuity",
      "lesson": "Maintain character position from previous clip.",
      "confidence": 0.88,
      "source_feedback_ids": ["fb-001"],
      "archived": false,
      "created_at": "2026-08-04T10:30:00Z"
    }
  ]
}
```

```bash
curl "http://127.0.0.1:8000/api/projects/demo/prompt-lessons?category=continuity"
```

---

#### `POST /api/projects/{project_name}/prompt-lessons`

Create a prompt learning lesson.

**Body (JSON):**
```json
{
  "scope": "project",
  "clip_key": null,
  "category": "continuity",
  "lesson": "Maintain character position across clip boundaries.",
  "source_feedback_ids": ["fb-001"],
  "confidence": 0.85,
  "positive_examples": [],
  "negative_examples": []
}
```

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `scope` | string | no | `"project"` | `"project"` or `"clip"` |
| `clip_key` | string | no | — | Clip key (required if `scope: "clip"`) |
| `category` | string | no | `"other"` | Lesson category |
| `lesson` | string | yes | — | The lesson text |
| `source_feedback_ids` | list[string] | no | `[]` | Feedback IDs that inspired this lesson |
| `confidence` | float | no | `0.8` | Confidence score (0–1) |
| `positive_examples` | list[string] | no | `[]` | Positive examples |
| `negative_examples` | list[string] | no | `[]` | Negative examples |

**Response 200:**
```json
{
  "lesson": {"id": "lesson-001", "scope": "project", "..."},
  "clip_state": { "..." }
}
```

**Errors:** 400 (validation error)

```bash
curl -X POST http://127.0.0.1:8000/api/projects/demo/prompt-lessons \
  -H "Content-Type: application/json" \
  -d '{"lesson": "Maintain character position.", "category": "continuity", "confidence": 0.85}'
```

---

#### `PATCH /api/projects/{project_name}/prompt-lessons/{lesson_id}`

Update a prompt lesson. Omitted fields are left unchanged.

All fields from the create request are updatable, plus:
| Field | Type | Description |
|---|---|---|
| `archived` | bool | Archive/unarchive the lesson |

**Response 200:**
```json
{
  "lesson": {"id": "lesson-001", "archived": true, "..."},
  "clip_state": { "..." }
}
```

**Errors:** 404 (lesson not found), 400 (validation error)

```bash
curl -X PATCH http://127.0.0.1:8000/api/projects/demo/prompt-lessons/lesson-001 \
  -H "Content-Type: application/json" \
  -d '{"archived": true}'
```

---

#### `GET /api/projects/{project_name}/prompt-eval-cases`

List prompt evaluation test cases.

**Query params:**
| Param | Type | Description |
|---|---|---|
| `clip_index` | int | Filter by clip index (optional) |
| `enabled` | bool | Filter by enabled status (optional) |

**Response 200:**
```json
{
  "schema_version": 1,
  "project_name": "demo",
  "clip_index": null,
  "enabled": null,
  "cases": [
    {
      "id": "ec-001",
      "name": "Continuity check",
      "input": {...},
      "expected_behavior": ["Character stays at left edge"],
      "failure_categories": ["continuity"],
      "enabled": true
    }
  ]
}
```

```bash
curl "http://127.0.0.1:8000/api/projects/demo/prompt-eval-cases?enabled=true"
```

---

#### `PATCH /api/projects/{project_name}/prompt-eval-cases/{case_id}`

Update an eval case. Omitted fields are left unchanged.

**Body (JSON):** All optional:
```json
{
  "name": "Updated name",
  "input": {"key": "value"},
  "expected_behavior": ["Behavior 1", "Behavior 2"],
  "failure_categories": ["continuity", "lighting"],
  "enabled": false
}
```

**Response 200:**
```json
{
  "eval_case": {"id": "ec-001", "enabled": false, "..."},
  "clip_state": { "..." }
}
```

**Errors:** 404 (case not found), 400 (validation error)

```bash
curl -X PATCH http://127.0.0.1:8000/api/projects/demo/prompt-eval-cases/ec-001 \
  -H "Content-Type: application/json" \
  -d '{"enabled": false}'
```

---

#### `POST /api/projects/{project_name}/prompts/{prompt_version_id}/revise-from-feedback`

Revise a prompt version using collected feedback, lessons, and eval cases. This is the core feedback-driven prompt improvement endpoint. It:
1. Collects matching feedback items for the prompt version
2. Retrieves relevant lessons and eval cases
3. Generates a revised prompt via AI
4. Runs quality and learning evaluation checks
5. Creates a new prompt version with full provenance

**Body (JSON):**
```json
{
  "clip_index": 2,
  "feedback_ids": ["fb-001", "fb-002"],
  "lesson_ids": ["lesson-001"],
  "provider": "openai"
}
```

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `clip_index` | int | yes | — | Clip index |
| `feedback_ids` | list[string] | no | `[]` | Specific feedback IDs to address (empty = all open negative feedback) |
| `lesson_ids` | list[string] | no | `[]` | Specific lesson IDs to apply (empty = auto-retrieved) |
| `provider` | string | no | `"openai"` | AI provider |

**Response 200:**
```json
{
  "prompt_version": {
    "prompt_version_id": "def456",
    "video_model_prompt": "Revised prompt text...",
    "explanation": "Updated character position to maintain continuity...",
    "quality_report": {...},
    "revision_source_prompt_version_id": "abc123",
    "revision_feedback_ids": ["fb-001"],
    "revision_lesson_ids": ["lesson-001"],
    "applied_prompt_lessons": [...],
    "applied_prompt_eval_cases": [...]
  },
  "quality_report": {...},
  "learning_report": {...},
  "clip_state": { "..." }
}
```

**Errors:** 404 (prompt version not found), 400 (empty prompt text, no matching feedback, bad feedback/lesson IDs), 500 (AI generation failure, empty revision output)

```bash
curl -X POST http://127.0.0.1:8000/api/projects/demo/prompts/abc123/revise-from-feedback \
  -H "Content-Type: application/json" \
  -d '{"clip_index": 2, "provider": "openai"}'
```

### Jobs & Workflow

Video generation, workflow execution, and job management. Jobs run as in-process background tasks — they do not survive a server restart.

---

#### `POST /api/projects/{project_name}/generate-video`

Generate a video for a clip (synchronous/blocking call — returns the result directly).

**Body (JSON):**
```json
{
  "clip_index": 2,
  "prompt_version_index": null,
  "provider": "segmind",
  "resolution": "720p",
  "generate_audio": false,
  "aspect_ratio": "9:16",
  "duration": 5
}
```

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `clip_index` | int | yes | — | Clip index |
| `prompt_version_index` | int | no | — | Prompt version index (null = latest) |
| `provider` | string | no | `"segmind"` | Video generation provider |
| `resolution` | string | no | `"720p"` | `"480p"`, `"720p"`, `"1080p"`, or `"4k"` |
| `generate_audio` | bool | no | `false` | Whether to generate audio |
| `aspect_ratio` | string | no | `"9:16"` | `"16:9"`, `"9:16"`, `"1:1"`, `"4:3"`, `"3:4"`, `"21:9"`, or `"adaptive"` |
| `duration` | int | no | `5` | Video duration in seconds |

**Response 200:** Generated video metadata and URLs.

```bash
curl -X POST http://127.0.0.1:8000/api/projects/demo/generate-video \
  -H "Content-Type: application/json" \
  -d '{"clip_index": 2, "resolution": "1080p", "aspect_ratio": "16:9", "duration": 8}'
```

---

#### `GET /api/projects/{project_name}/jobs`

List recent jobs for a project.

**Response 200:**
```json
{
  "project_name": "demo",
  "jobs": [
    {
      "id": "job-001",
      "type": "workflow",
      "status": "completed",
      "project_name": "demo",
      "provider": "openai",
      "created_at": "2026-08-04T10:30:00Z",
      "finished_at": "2026-08-04T10:32:00Z"
    }
  ]
}
```

```bash
curl http://127.0.0.1:8000/api/projects/demo/jobs
```

---

#### `GET /api/projects/{project_name}/jobs/{job_id}`

Get a single job with its full record including logs, result, and error details.

**Response 200:**
```json
{
  "id": "job-001",
  "type": "generate_video",
  "status": "succeeded",
  "project_name": "demo",
  "clip_index": 2,
  "provider": "segmind",
  "payload": {...},
  "logs": [...],
  "result": {"video_url": "...", "..."},
  "error": null,
  "created_at": "2026-08-04T10:30:00Z",
  "started_at": "2026-08-04T10:30:01Z",
  "finished_at": "2026-08-04T10:32:00Z"
}
```

Job statuses: `queued` → `running` → `succeeded` / `failed` / `cancelled`. A `cancel_requested` flag triggers transition to `cancelling`.

```bash
curl http://127.0.0.1:8000/api/projects/demo/jobs/job-001
```

---

#### `POST /api/projects/{project_name}/jobs/{job_id}/cancel`

Request cancellation of a running job. If the job has already finished, returns it unchanged.

**Response 200:**
```json
{
  "id": "job-001",
  "status": "cancelling",
  "cancel_requested": true,
  "..."
}
```

```bash
curl -X POST http://127.0.0.1:8000/api/projects/demo/jobs/job-001/cancel
```

---

#### `POST /api/projects/{project_name}/jobs/workflow`

Create and enqueue a workflow job (AI prompt generation from feedback). The job is created with status `queued` and executed via `BackgroundTasks`.

**Body (JSON):**
```json
{
  "feedback_index": 0,
  "provider": "openai",
  "agent_run_id": null
}
```

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `feedback_index` | int | yes | — | Feedback group index to run workflow on |
| `provider` | string | no | `"openai"` | AI provider |
| `agent_run_id` | string | no | — | Associate with an existing agent run |

**Response 200:** The created job object with `status: "queued"`.

```bash
curl -X POST http://127.0.0.1:8000/api/projects/demo/jobs/workflow \
  -H "Content-Type: application/json" \
  -d '{"feedback_index": 0, "provider": "gemini"}'
```

---

#### `POST /api/projects/{project_name}/jobs/generate-video`

Create and enqueue a video generation job. The job is created with status `queued` and executed via `BackgroundTasks`.

**Body (JSON):** Same schema as [`POST /generate-video`](#post-apiprojectsproject_namegenerate-video):
```json
{
  "clip_index": 2,
  "prompt_version_index": null,
  "provider": "segmind",
  "resolution": "720p",
  "generate_audio": false,
  "aspect_ratio": "9:16",
  "duration": 5
}
```

**Response 200:** The created job object with `status: "queued"`.

```bash
curl -X POST http://127.0.0.1:8000/api/projects/demo/jobs/generate-video \
  -H "Content-Type: application/json" \
  -d '{"clip_index": 2, "resolution": "1080p", "aspect_ratio": "16:9"}'
```

---

#### `GET /api/run-workflow`

Run a workflow synchronously with Server-Sent Events streaming. Spawns `main.py` as a subprocess and streams stdout/stderr line-by-line as SSE events.

> ⚠️ **Streaming endpoint** — returns `text/event-stream`, not JSON. See the [SSE section](#server-sent-events-sse) for details.

> This endpoint is not project-scoped in its URL path (unlike most other endpoints).

**Query params:**
| Param | Type | Required | Default | Description |
|---|---|---|---|---|
| `project` | string | yes | — | Project name |
| `index` | int | yes | — | Feedback index to run workflow on |
| `provider` | string | no | `"openai"` | AI provider |

**Response:** `text/event-stream`

**Errors:** 500 (raw feedback mapping failed), 404 (assets not found for project)

```bash
curl -N "http://127.0.0.1:8000/api/run-workflow?project=demo&index=0&provider=openai"
```

---

#### `GET /api/workflow-result`

Get the output of the most recent workflow run for a project. Reads from `{project_data_dir}/output.json`.

> This endpoint is not project-scoped in its URL path (unlike most other endpoints).

**Query params:**
| Param | Type | Required | Description |
|---|---|---|---|
| `project` | string | yes | Project name |

**Response 200:** The contents of `output.json` from the last workflow run.

**Errors:** 404 (no output.json — run a workflow first), 500 (JSON parse error)

```bash
curl "http://127.0.0.1:8000/api/workflow-result?project=demo"
```

### Clip Chat

Per-clip AI chat with tool use, agent runs, and persistent memory.

---

#### `GET /api/projects/{project_name}/chat/clip/{clip_index}`

Get the chat history and context for a clip. Returns existing messages, memory items, agent runs, and a context summary.

**Response 200:**
```json
{
  "project_name": "demo",
  "clip_index": 2,
  "clip_key": "clip_002",
  "messages": [
    {"role": "assistant", "content": "I am attached to clip #3...", "metadata": {"kind": "welcome"}}
  ],
  "memory": [...],
  "agent_runs": [...],
  "context": {
    "project": {...},
    "clip": {...},
    "clip_state": {...},
    "learning_state": {...},
    "adjacent_clips": [...],
    "feedback_count": 3,
    "selected_asset_count": 5,
    "prompt_ready": true
  }
}
```

On first access, a welcome message is automatically created and saved.

```bash
curl http://127.0.0.1:8000/api/projects/demo/chat/clip/2
```

---

#### `POST /api/projects/{project_name}/chat/clip`

Send a message to the clip chat. The backend processes the message through the AI provider, runs any triggered tools, auto-extracts and saves memory notes, infers suggested actions, and manages agent runs and background jobs.

**Body (JSON):**
```json
{
  "clip_index": 2,
  "message": "What references do we have for this clip's lighting style?",
  "provider": "openai"
}
```

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `clip_index` | int | yes | — | Clip index |
| `message` | string | yes | — | User's chat message (non-empty) |
| `provider` | string | no | `"openai"` | AI provider |

**Response 200:** Chat messages, assistant reply, memory updates, suggested actions, and agent run state:
```json
{
  "project_name": "demo",
  "clip_index": 2,
  "clip_key": "clip_002",
  "messages": [...],
  "assistant_message": {
    "role": "assistant",
    "content": "You have 5 reference assets for this clip...",
    "metadata": {"provider": "openai", "actions": [...], "..."}
  },
  "memory": [...],
  "suggested_actions": [...],
  "agent_run": {...},
  "clip_state": {...}
}
```

**Errors:** 400 (empty message)

```bash
curl -X POST http://127.0.0.1:8000/api/projects/demo/chat/clip \
  -H "Content-Type: application/json" \
  -d '{"clip_index": 2, "message": "What references do we have for lighting?", "provider": "openai"}'
```

---

#### `POST /api/projects/{project_name}/chat/clip/action`

Execute a specific chat action (from the prompt learning action types). Actions are predefined tool operations like generating summaries or applying lessons. The action is ran as a one-step agent run.

**Body (JSON):**
```json
{
  "clip_index": 2,
  "action": {"type": "regenerate_clip_summary", "label": "Re-analyze clip"},
  "message": "Re-analyze this clip with fresh context.",
  "provider": "openai"
}
```

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `clip_index` | int | yes | — | Clip index |
| `action` | dict | yes | — | Action object with `type` key |
| `message` | string | no | `""` | Optional context message |
| `provider` | string | no | `"openai"` | AI provider |

**Response 200:**
```json
{
  "project_name": "demo",
  "clip_index": 2,
  "clip_key": "clip_002",
  "action_result": {"status": "ok", "message": "Clip summary regenerated."},
  "agent_run": {...},
  "clip_state": {...}
}
```

**Errors:** 400 (action type not in allowed prompt learning actions)

```bash
curl -X POST http://127.0.0.1:8000/api/projects/demo/chat/clip/action \
  -H "Content-Type: application/json" \
  -d '{"clip_index": 2, "action": {"type": "regenerate_clip_summary"}, "provider": "openai"}'
```

---

#### `POST /api/projects/{project_name}/chat/memory`

Manually save a memory item associated with a clip or project.

**Body (JSON):**
```json
{
  "clip_index": 2,
  "text": "Client prefers warm color grading for all outdoor scenes.",
  "scope": "project"
}
```

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `clip_index` | int | yes | — | Clip index (for context resolution) |
| `text` | string | yes | — | Memory text (non-empty) |
| `scope` | string | no | `"clip"` | `"clip"` or `"project"` |

**Response 200:**
```json
{
  "memory": [...],
  "item": {
    "id": "mem-001",
    "text": "Client prefers warm color grading for all outdoor scenes.",
    "scope": "project",
    "source": "manual",
    "confidence": 0.9
  }
}
```

**Errors:** 400 (empty text)

```bash
curl -X POST http://127.0.0.1:8000/api/projects/demo/chat/memory \
  -H "Content-Type: application/json" \
  -d '{"clip_index": 2, "text": "Client prefers warm color grading.", "scope": "project"}'
```

## Server-Sent Events (SSE)

The `GET /api/run-workflow` endpoint returns `text/event-stream` — a streaming SSE response, not JSON.

### Event Format

Each line is framed as `data: <content>\n\n` (standard SSE). The server sends these event types:

| Prefix | Meaning |
|---|---|
| `[START]` | Workflow subprocess is launching |
| `[CONTEXT]` | Continuity reference context (previous clip last-frame) |
| *(no prefix)* | Subprocess stdout line |
| `[STDERR]` | Subprocess stderr line |
| `[SUCCESS]` | Workflow completed successfully |
| `[ERROR]` | Workflow failed or subprocess crashed |

### Example Stream

```
data: [START] Launching workflow subprocess for feedback index 0...

data: Executing command: python main.py --index 0 --feedback-path ...
data: [Agent] Loading project timeline...
data: [Agent] Generating plan for clip #1...
data: [Agent] Generating video prompt...

data: [SUCCESS] Workflow execution finished with exit code 0
```

### Usage

```bash
# Stream the workflow output in real time (-N disables buffering)
curl -N "http://127.0.0.1:8000/api/run-workflow?project=demo&index=0&provider=openai"

# Read the result after completion
curl "http://127.0.0.1:8000/api/workflow-result?project=demo"
```

### Behavior Notes

- Spawns `main.py` as a **subprocess** — the workflow CLI tool, not a thread
- The process inherits the current environment (API keys, storage paths)
- If the subprocess exits with code 0, output is automatically saved to prompt versions
- Long-running; typical workflows take 30 seconds to several minutes depending on provider and clip count
- The server continues accepting other requests while streaming

## Static File Mounts

Two directories are mounted as static file servers:

| Mount | Directory | Purpose |
|---|---|---|
| `/assets` | `{storage}/assets/` | Uploaded and extracted asset files |
| `/data` | `{storage}/data/` | Generated data: filmstrips, logs, output files |

URLs returned in API responses (e.g., filmstrip frame URLs, asset paths) are relative to these mounts. Access them directly:

```bash
curl http://127.0.0.1:8000/data/demo/timeline_filmstrips/0002_Clip_02/frame_01.jpg
```

> ⚠️ These mounts expose the entire storage tree over HTTP with no authentication. The server is designed for localhost use only.

## OpenAPI Docs

FastAPI auto-generates interactive API documentation while the server is running:

- **Swagger UI:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc:** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
- **OpenAPI spec:** [http://127.0.0.1:8000/openapi.json](http://127.0.0.1:8000/openapi.json)

These are the schema ground truth — the `/docs` UI shows request/response schemas with live "Try it out" functionality. The OpenAPI spec at `/openapi.json` is the canonical machine-readable contract.

## Testing

Run the backend test suite:

```bash
cd backend
source .venv/bin/activate
pytest
```

Tests exercise all router endpoints. See `tests/` for the full suite (~26 test files).

## Security & Operational Caveats

1. **No authentication.** Only `POST /shutdown` checks the `x-loka-shutdown-token` header. Every other endpoint is open. CORS is set to `allow_origins=["*"]` with `allow_credentials=True`. This server is designed for **localhost use by the Tauri desktop app** — do not expose it to a network.

2. **`POST /api/config/secrets` is unauthenticated.** It writes provider API keys to a chmod 600 config file. Keys that are set via real OS environment variables are locked (returns 409) and cannot be overwritten through the API.

3. **`GET /api/run-workflow` is SSE streaming.** It returns `text/event-stream`, not JSON. The endpoint spawns `main.py` as a subprocess and streams stdout/stderr. Use `curl -N` to consume it without buffering.

4. **Background jobs are in-process.** Jobs created via `/jobs/workflow` and `/jobs/generate-video` use FastAPI `BackgroundTasks`. They do **not** survive a server restart. There is no persistent job queue.

5. **No response model validation.** Endpoints return raw Python dicts — no `response_model` Pydantic validation on responses. The OpenAPI spec at `/openapi.json` reflects the actual return shapes.

6. **Schema duplication.** Some Pydantic models are defined in both `src/schema.py` and inline in router files. The **router-inline definitions are authoritative** for what each endpoint accepts. Feedback schemas live in `src/prompt_feedback.py`. When in doubt, check the router file directly.

7. **Storage is JSON files.** There is no database ORM. Project data is stored as JSON files in `{storage}/data/{project}/`. An optional SQLite database at `{storage}/data/asset_url_cache.sqlite3` caches asset URLs (Supabase alternative available via `LOKA_ASSET_CACHE_BACKEND`).

## Known Inconsistencies

- **Singular `project` URL:** `GET /api/project/{project_name}` uses the singular `project`, while all other project-scoped endpoints use the plural `projects`. This is a legacy inconsistency.
- **Duplicate job endpoints:** `POST /api/projects/{project_name}/generate-video` (sync) and `POST /api/projects/{project_name}/jobs/generate-video` (async job) both generate video — the former blocks, the latter returns a job record immediately.
- **Non-project-scoped workflow endpoints:** `/api/run-workflow` and `/api/workflow-result` take a `project` query parameter instead of using the URL path convention.

## Related Docs

- [Root README](../README.md) — full project overview, frontend, Tauri packaging
- [Server Refactor Design](docs/superpowers/specs/2026-08-03-server-refactor-design.md) — architecture rationale for the router split
- [Seedance Migration](segmind_seedance_migration.md) — Seedance 2.0 video generation flow
- [Video Generation Skill](skill/video_generation_skill.md) — Claude skill for Seedance prompting
