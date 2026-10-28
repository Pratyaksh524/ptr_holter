from __future__ import annotations

import os
import sys
from pathlib import Path


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def should_use_runtime_dir() -> bool:
    """
    True when we should store runtime-created files outside the install folder.

    - Always True for PyInstaller builds (sys.frozen).
    - Can be forced for dev via ECG_FORCE_RUNTIME_DIR=1.
    """
    if is_frozen():
        return True
    flag = str(os.getenv("ECG_FORCE_RUNTIME_DIR", "0")).strip().lower()
    return flag in {"1", "true", "yes", "on"}


def runtime_dir() -> Path:
    """
    Writable per-user directory for app runtime data.

    Uses ECG_RUNTIME_DIR if set, otherwise LOCALAPPDATA\\Deckmount\\Comprehensive.
    """
    base_dir = os.getenv("ECG_RUNTIME_DIR", "").strip()
    if base_dir:
        p = Path(base_dir).expanduser()
    else:
        local_appdata = os.getenv("LOCALAPPDATA") or str(Path.home())
        p = Path(local_appdata) / "Deckmount" / "Comprehensive"
    p.mkdir(parents=True, exist_ok=True)
    return p


def data_file(filename: str) -> Path:
    """
    Path for runtime-created JSON/support files.

    In frozen builds this keeps files out of the EXE folder (and out of _MEI temp),
    while remaining writable for normal users.
    """
    if should_use_runtime_dir():
        return runtime_dir() / filename
    # Keep dev-time JSON files anchored to the project root so imports do not
    # depend on the current working directory.
    return Path(__file__).resolve().parents[2] / filename

