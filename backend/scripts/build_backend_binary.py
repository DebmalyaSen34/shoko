from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> int:
    backend_root = Path(__file__).resolve().parents[1]
    dist_dir = backend_root / "dist"
    build_dir = backend_root / "build"
    pyinstaller_config_dir = build_dir / "pyinstaller-config"

    shutil.rmtree(dist_dir / "backend-server", ignore_errors=True)
    pyinstaller_config_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onedir",
        "--name",
        "backend-server",
        "--paths",
        str(backend_root),
        "--distpath",
        str(dist_dir),
        "--workpath",
        str(build_dir),
        "--specpath",
        str(build_dir),
        "--add-data",
        f"{backend_root / 'config'}{os.pathsep}config",
        "--add-data",
        f"{backend_root / 'skill'}{os.pathsep}skill",
        "--hidden-import",
        "uvicorn.logging",
        "--hidden-import",
        "uvicorn.loops.auto",
        "--hidden-import",
        "uvicorn.protocols.http.auto",
        "--hidden-import",
        "uvicorn.protocols.websockets.auto",
        "--hidden-import",
        "uvicorn.lifespan.on",
        str(backend_root / "backend_server.py"),
    ]

    env = os.environ.copy()
    env["PYINSTALLER_CONFIG_DIR"] = str(pyinstaller_config_dir)

    subprocess.run(command, check=True, env=env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
