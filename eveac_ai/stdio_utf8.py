"""Force UTF-8 stdio so Cursor/VS Code terminals do not GBK-mojibake Chinese."""

from __future__ import annotations

import io
import os
import sys


def configure_utf8_stdio() -> None:
    os.environ["PYTHONUTF8"] = "1"
    os.environ["PYTHONIOENCODING"] = "utf-8"
    if sys.platform == "win32":
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            kernel32.SetConsoleOutputCP(65001)
            kernel32.SetConsoleCP(65001)
        except (AttributeError, OSError):
            pass
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if stream is None:
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace", newline="\n", line_buffering=True)
            continue
        except (AttributeError, OSError, ValueError):
            pass
        buf = getattr(stream, "buffer", None)
        if buf is None:
            continue
        wrapped = io.TextIOWrapper(
            buf,
            encoding="utf-8",
            errors="replace",
            newline="\n",
            line_buffering=True,
            write_through=True,
        )
        setattr(sys, name, wrapped)
