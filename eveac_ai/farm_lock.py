"""Single-process farm lock so multiple match20 instances do not share one GPU."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    except AttributeError:
        # Windows without kill(0): fall back to OpenProcess via ctypes is heavy; try tasklist-ish.
        try:
            import ctypes

            k = ctypes.windll.kernel32  # type: ignore[attr-defined]
            h = k.OpenProcess(0x1000, False, pid)
            if h:
                k.CloseHandle(h)
                return True
            return False
        except Exception:
            return False
    return True


def acquire_farm_lock(samples: Path) -> Path | None:
    """Write samples/farm.lock. Returns lock path, or None if another live farm holds it."""
    samples.mkdir(parents=True, exist_ok=True)
    path = samples / "farm.lock"
    if path.is_file():
        try:
            blob = json.loads(path.read_text(encoding="utf-8")) or {}
            old = int(blob.get("pid") or 0)
        except (json.JSONDecodeError, TypeError, ValueError):
            old = 0
        if old and old != os.getpid() and _pid_alive(old):
            print(f"farm.lock held by live pid={old}; refuse second farm process", flush=True)
            return None
    blob: dict[str, Any] = {
        "pid": os.getpid(),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    path.write_text(json.dumps(blob, indent=2) + "\n", encoding="utf-8")
    return path


def release_farm_lock(path: Path | None) -> None:
    if path is None:
        return
    try:
        if path.is_file():
            blob = json.loads(path.read_text(encoding="utf-8")) or {}
            if int(blob.get("pid") or 0) == os.getpid():
                path.unlink(missing_ok=True)
    except Exception:
        pass


def consume_stop_file(samples: Path) -> bool:
    """True if samples/farm.stop was present (then deleted). Windows-safe stop; do not SIGTERM."""
    path = Path(samples) / "farm.stop"
    if not path.is_file():
        return False
    try:
        path.unlink()
    except OSError:
        pass
    return True
