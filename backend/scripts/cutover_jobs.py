#!/usr/bin/env python3
"""Cutover script for the jobs router.

For each route owned by src/routers/jobs.py:
  1. Extract the handler (decorator + def + body) from server.py and from the
     router module.
  2. Compare bodies semantically (whitespace-normalized); refuse to delete if
     they differ.
  3. Delete the handler from server.py.

Usage: .venv/bin/python scripts/cutover_projects.py
"""

import re
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

SERVER = BACKEND_DIR / "server.py"
ROUTER = BACKEND_DIR / "src" / "routers" / "jobs.py"

DELETE_ROUTES = {
    ("POST", "/api/projects/{project_name}/generate-video"),
    ("GET", "/api/projects/{project_name}/jobs"),
    ("GET", "/api/projects/{project_name}/jobs/{job_id}"),
    ("POST", "/api/projects/{project_name}/jobs/{job_id}/cancel"),
    ("POST", "/api/projects/{project_name}/jobs/workflow"),
    ("POST", "/api/projects/{project_name}/jobs/generate-video"),
    ("GET", "/api/run-workflow"),
    ("GET", "/api/workflow-result"),
}

DECORATOR_RE = re.compile(r'^@(?:app|router)\.(get|post|patch|put|delete)\("([^"]+)"\)')
DEF_RE = re.compile(r"^(?:async )?def (\w+)\(")


def extract_handlers(path: Path) -> dict[tuple[str, str], dict]:
    """Map (method, path) -> {"decorator": ..., "body": [...], "end": idx}."""
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    handlers = {}
    i = 0
    n = len(lines)
    while i < n:
        match = DECORATOR_RE.match(lines[i].strip())
        if match:
            method, route_path = match.group(1).upper(), match.group(2)
            decorator = lines[i]
            j = i + 1
            while j < n and not lines[j].strip():
                j += 1
            def_match = DEF_RE.match(lines[j].strip()) if j < n else None
            if not def_match:
                i += 1
                continue
            body = [lines[j]]
            # Consume a multiline signature through its closing '):' (which may
            # sit at column 0) before scanning the indented body.
            sig_end = j
            while sig_end < n and not lines[sig_end].rstrip().endswith("):"):
                sig_end += 1
            if sig_end > j:
                body.extend(lines[j + 1 : sig_end + 1])
            k = sig_end + 1
            while k < n and (not lines[k].strip() or lines[k][0] in " \t"):
                body.append(lines[k])
                k += 1
            handlers[(method, route_path)] = {"decorator": decorator, "body": body, "end": k}
            i = k
        else:
            i += 1
    return handlers


def normalized(body_lines):
    """Whitespace-, comment-, and import-insensitive comparison (comments and
    function-level imports don't affect behavior: moved copies may have
    dropped comments, and routers import from src modules instead of defining
    inline)."""
    return "\n".join(
        line.strip()
        for line in body_lines
        if line.strip()
        and not line.strip().startswith("#")
        and not line.strip().startswith(("from ", "import "))
    )


def extract_function(path: Path, name: str) -> list[str]:
    """Extract a top-level def block (without decorators) by function name."""
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    for i, line in enumerate(lines):
        match = DEF_RE.match(line.strip())
        if match and match.group(1) == name:
            body = [line]
            j = i + 1
            while j < len(lines) and (not lines[j].strip() or lines[j][0] in " \t"):
                body.append(lines[j])
                j += 1
            return body
    return []


def main():
    server_handlers = extract_handlers(SERVER)
    router_handlers = extract_handlers(ROUTER)

    found = 0
    for route in sorted(DELETE_ROUTES):
        server_entry = server_handlers.get(route)
        if not server_entry:
            print(f"SKIP (not found in server.py): {route}")
            continue
        found += 1
        router_entry = router_handlers.get(route)
        if not router_entry:
            print(f"ERROR: route exists in server.py but not in router: {route}")
            return 1
        match = normalized(server_entry["body"]) == normalized(router_entry["body"])
        print(f"OK: {route[0]} {route[1]} ({len(server_entry['body'])} body lines)")
        if not match:
            print(f"DIFFERS (refusing to delete): {route}")
            print("  server:", normalized(server_entry["body"])[:200])
            print("  router:", normalized(router_entry["body"])[:200])
            return 1

    if found != len(DELETE_ROUTES):
        print(f"ERROR: found {found}/{len(DELETE_ROUTES)} routes")
        return 1

    # Delete the handlers from server.py in one pass.
    lines = SERVER.read_text(encoding="utf-8").splitlines(keepends=True)
    drop = set()
    for route in DELETE_ROUTES:
        entry = server_handlers[route]
        decorator_line = entry["decorator"]
        for idx, line in enumerate(lines):
            if line == decorator_line and idx not in drop:
                drop.update(range(idx, entry["end"]))
                break

    new_lines = [line for idx, line in enumerate(lines) if idx not in drop]
    SERVER.write_text("".join(new_lines), encoding="utf-8")
    print(f"\nDeleted {found} handlers from server.py (server.py: {len(lines)} -> {len(new_lines)} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
