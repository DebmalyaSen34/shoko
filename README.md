# Loka15 Studio

Loka15 Studio is a desktop production tool for turning Adobe Premiere Pro project timelines and client feedback into structured AI video-generation plans. The app helps a creative team ingest a project package, organize reference assets, parse feedback, align comments to clips, generate Seedance-ready prompts, and inspect the resulting timeline, prompts, reference frames, audio trims, and quality notes.

The desktop app is branded as **Shoko** in the Tauri configuration.

## What It Does

- Creates project workspaces with a predictable asset folder structure.
- Imports Premiere Pro project packages and extracts timeline data from `.prproj` files.
- Accepts uploaded or manually entered feedback and aligns remarks to timeline clips.
- Organizes visual and audio reference assets by category.
- Generates clip-level plans and video prompts with OpenAI, Gemini, and Segmind-backed workflows.
- Preserves prompt versions, selected references, initial-frame prompts, quality reports, and generated media metadata.
- Provides a React timeline UI with asset browsing, clip chat memory, prompt preview, workflow logs, and runtime API-key settings.
- Runs as either a browser-based development app or a packaged Tauri desktop app with a managed local backend.

## Project Structure

```text
.
|-- src/                    # React + TypeScript frontend
|   |-- components/          # Timeline, asset browser, modals, chat, toasts
|   |-- lib/                 # API URL and formatting helpers
|   `-- types.ts             # Frontend data contracts
|-- backend/                 # FastAPI backend and AI workflow code
|   |-- config/              # Runtime settings and prompt templates
|   |-- scripts/             # Backend binary and media generation helpers
|   |-- skill/               # Seedance video-generation skill instructions
|   |-- src/                 # Parsers, generators, workflows, storage helpers
|   `-- tests/               # Pytest coverage for backend workflows
|-- src-tauri/               # Tauri desktop shell and backend process manager
|-- public/                  # Static Vite assets
`-- package.json             # Frontend and Tauri commands
```

## Requirements

- Node.js compatible with Vite 7
- Bun, used by the Tauri dev/build hooks
- Python 3.10+
- Rust and Cargo for Tauri
- System media tooling required by the backend workflows, especially `ffmpeg`/`ffprobe`
- API keys for the providers you plan to use:
  - `OPENAI_API_KEY`
  - `GEMINI_API_KEY`
  - `SEGMIND_API_KEY`

## Setup

Install frontend dependencies:

```bash
bun install
```

Create a backend virtual environment and install Python dependencies:

```bash
cd backend
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
cd ..
```

Create local environment files from the examples:

```bash
cp .env.example .env
cp backend/.env.example backend/.env
```

Then fill in provider keys in `backend/.env` as needed. The frontend `.env` is only for build-time Vite values; provider secrets belong in the backend environment or the app's runtime settings screen.

## Running Locally

### Browser Development

Start the FastAPI backend:

```bash
cd backend
. .venv/bin/activate
python -m uvicorn server:app --host 127.0.0.1 --port 8000 --reload
```

In another terminal, start Vite:

```bash
bun run dev
```

Open the Vite URL printed by the command. By default, the frontend talks to `http://127.0.0.1:8000`. To use another backend URL, set `VITE_API_BASE_URL` in `.env`.

### Tauri Desktop Development

```bash
bun run tauri dev
```

In desktop mode, React asks Tauri for the backend URL at runtime. Tauri chooses a free localhost port and starts the backend automatically. During development it looks for `backend/.venv/bin/python` first, then falls back to `python3`.

You can override backend startup with:

- `LOKA_BACKEND_COMMAND` - run a custom backend command.
- `LOKA_BACKEND_BINARY` - run a specific backend binary.
- `LOKA_BACKEND_PYTHON` - choose the Python executable for the development backend.

## Runtime Configuration

The backend loads configuration in this precedence order:

1. Real OS environment variables.
2. `LOKA_ENV_FILE`, when set.
3. Installed-app config at `{app storage}/config/.env`.
4. Local development file at `backend/.env`.
5. Code defaults.

Default app storage paths:

- macOS: `~/Library/Application Support/Loka15 Studio`
- Windows: `%LOCALAPPDATA%\Loka15 Studio`
- Linux: `${XDG_DATA_HOME:-~/.local/share}/Loka15 Studio`

Set `LOKA_STORAGE_DIR` as a real OS environment variable to override the storage root. The backend exposes `/api/config/runtime` and `/api/config/secrets` so the app can show key status and update non-OS-controlled secrets without returning secret values to the frontend.

## Project Data Model

Each project has separate data and asset roots under app storage:

```text
{storage}/
|-- data/{project}/
`-- assets/{project}/
    |-- 00_style/
    |-- 01_characters/
    |-- 02_props/
    |-- 03_locations/
    |-- 04_audio/
    |-- 05_references/
    `-- 06_clips/
        |-- _final/
        `-- _raw/
```

Premiere package imports create this structure, move `.prproj` files into the project root, place source video in `06_clips/_raw`, and place audio in `04_audio`.

## API Overview

The FastAPI backend is served from `backend/server.py`.

Key routes:

- `GET /health` and `GET /version` - backend lifecycle checks.
- `GET /api/projects` - list projects.
- `POST /api/projects` - create a project.
- `POST /api/projects/{project_name}/premiere-package` - upload and unpack a Premiere project zip.
- `POST /api/projects/{project_name}/assets/{category}` - upload assets into a known category.
- `POST /api/projects/{project_name}/feedback` - upload feedback text/file.
- `POST /api/projects/{project_name}/feedback/item` - add one manual feedback item.
- `GET /api/project/{project_name}` - load timeline, assets, feedback, prompt history, and derived artifacts.
- `GET /api/run-workflow` - stream workflow logs with Server-Sent Events.
- `GET /api/workflow-result` - read workflow output.
- `GET/POST /api/projects/{project_name}/chat/...` - clip chat and memory endpoints.
- `GET/POST /api/config/...` - runtime configuration and secret status.

## Testing

Run the backend test suite from the backend directory:

```bash
cd backend
. .venv/bin/activate
pytest
```

Build the frontend type-check and production bundle:

```bash
bun run build
```

## Packaging

The packaged Tauri app expects a backend executable named `backend-server` inside the app resources or alongside the app executable. Build it with PyInstaller:

```bash
cd backend
. .venv/bin/activate
pip install -r requirements-build.txt
python scripts/build_backend_binary.py
cd ..
```

Then build the Tauri app:

```bash
bun run tauri build
```

The Tauri config also includes `src-tauri/resources/backend/` as bundled resources. Keep that directory and the packaged backend layout in sync with the release process.

## Common Workflows

1. Create or select a project in the app.
2. Upload a Premiere project package, or add assets manually into style, character, prop, location, audio, reference, and clip categories.
3. Upload feedback or add a manual feedback item.
4. Choose the AI provider in the UI.
5. Run the workflow for a feedback item.
6. Review the generated prompt, initial-frame prompt, references, audio trim, quality report, and version history.
7. Copy the final prompt or continue iterating through clip chat and prompt regeneration.

## Troubleshooting

- **Frontend cannot connect to the backend:** start `backend/server.py` on `127.0.0.1:8000`, or set `VITE_API_BASE_URL` to the correct URL.
- **Tauri window opens but no data loads:** make sure `backend/.venv` exists and contains the backend dependencies, or set `LOKA_BACKEND_PYTHON`.
- **Provider calls fail:** check `/api/config/runtime` or the Settings modal to confirm the relevant API key is configured.
- **Storage appears empty:** confirm whether `LOKA_STORAGE_DIR` is set in the OS environment; installed-app storage and local repo files are intentionally separate.
- **Media extraction fails:** verify `ffmpeg` and `ffprobe` are installed and available on `PATH`.
- **Packaged backend does not start:** build the PyInstaller binary and confirm Tauri can find `backend-server` in the resource directory or via `LOKA_BACKEND_BINARY`.
