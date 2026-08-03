#!/usr/bin/env python3
"""Semantic diff: every top-level function in server.py vs the same-named
function in any src/ module. Reports body differences (ignoring whitespace,
comments, and import lines) for the remaining inline copies.

Usage: .venv/bin/python scripts/diff_all_functions.py
"""

import re
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

DEF_RE = re.compile(r"^(?:async )?def (\w+)\(")


def extract_functions(path: Path) -> dict[str, list[str]]:
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    funcs = {}
    i = 0
    n = len(lines)
    while i < n:
        match = DEF_RE.match(lines[i].strip())
        if match:
            name = match.group(1)
            # Skip methods (indented defs are not matched since we only look
            # at column-0 lines); this pass only finds top-level defs.
            body = [lines[i]]
            # Consume multiline signature
            sig_end = i
            while sig_end < n and not lines[sig_end].rstrip().endswith("):"):
                sig_end += 1
            if sig_end > i:
                body.extend(lines[i + 1 : sig_end + 1])
            k = sig_end + 1
            while k < n and (not lines[k].strip() or lines[k][0] in " \t"):
                body.append(lines[k])
                k += 1
            funcs[name] = body
            i = k
        else:
            i += 1
    return funcs


def normalized(body_lines):
    return "\n".join(
        line.strip()
        for line in body_lines
        if line.strip()
        and not line.strip().startswith("#")
        and not line.strip().startswith(("from ", "import "))
    )


def main():
    server_funcs = extract_functions(BACKEND_DIR / "server.py")
    src_funcs = {}
    for path in sorted((BACKEND_DIR / "src").rglob("*.py")):
        for name, body in extract_functions(path).items():
            src_funcs.setdefault(name, []).append((path, body))

    diffs = 0
    for name in sorted(server_funcs):
        candidates = src_funcs.get(name)
        if not candidates:
            continue
        srv_norm = normalized(server_funcs[name])
        matching = [c for c in candidates if normalized(c[1]) == srv_norm]
        if matching:
            continue
        for path, body in candidates:
            if normalized(body) != srv_norm:
                diffs += 1
                print(f"DIFF {name} (server.py vs {path.relative_to(BACKEND_DIR)})")
    print(f"\n{diffs} divergences found")
    return 1 if diffs else 0


if __name__ == "__main__":
    sys.exit(main())
