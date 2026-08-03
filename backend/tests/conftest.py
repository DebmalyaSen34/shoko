import sys
from pathlib import Path

import server


def patch_storage_dirs(monkeypatch, data_dir, assets_dir=None):
    """Point DATA_DIR/ASSETS_DIR at throwaway dirs in every module holding a
    binding.

    The refactor moved the canonical paths into src.storage_paths, and modules
    that did `from src.storage_paths import DATA_DIR` hold their own binding.
    Patching only `server.DATA_DIR` (as the pre-refactor tests did) leaves the
    moved functions reading the real storage dir. Sweeping sys.modules covers
    every importer plus the remaining inline server.py code.
    """
    if assets_dir is None:
        assets_dir = Path(data_dir).parent / "assets"
    patched = set()
    for module in list(sys.modules.values()):
        for name, value in (("DATA_DIR", data_dir), ("ASSETS_DIR", assets_dir)):
            if hasattr(module, name):
                try:
                    setattr(module, name, value)
                    patched.add(f"{module.__name__}.{name}")
                except (AttributeError, TypeError):
                    pass
    return patched


def patch_storage_dirs_manual(data_dir, assets_dir=None):
    """Same sweep as patch_storage_dirs but with a manual restore function
    (for unittest-style tests that don't have a pytest monkeypatch fixture)."""
    if assets_dir is None:
        assets_dir = Path(data_dir).parent / "assets"
    saved = []
    for module in list(sys.modules.values()):
        for name, value in (("DATA_DIR", data_dir), ("ASSETS_DIR", assets_dir)):
            if hasattr(module, name):
                try:
                    old = getattr(module, name)
                    setattr(module, name, value)
                    saved.append((module, name, old))
                except (AttributeError, TypeError):
                    pass

    def restore():
        for module, name, old in saved:
            try:
                setattr(module, name, old)
            except (AttributeError, TypeError):
                pass

    return restore
