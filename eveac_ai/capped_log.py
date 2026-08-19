"""Size-capped UTF-8 log. Each file ≤ 500MB; overflow rotates (drop older backup)."""

from __future__ import annotations

from pathlib import Path

LOG_CAP_BYTES = 500 * 1024 * 1024


class CappedLog:
    def __init__(self, path: Path, *, cap: int = LOG_CAP_BYTES, kind: str = "log") -> None:
        self.path = Path(path)
        self.cap = int(cap)
        self.kind = kind
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = self.path.open("a", encoding="utf-8")

    def write(self, text: str) -> None:
        if not text:
            return
        if not text.endswith("\n"):
            text += "\n"
        blob = text.encode("utf-8")
        try:
            self._fp.flush()
            sz = self.path.stat().st_size if self.path.is_file() else 0
            if sz + len(blob) > self.cap:
                self._rotate()
            self._fp.write(text)
            self._fp.flush()
        except OSError:
            return

    def flush(self) -> None:
        try:
            self._fp.flush()
        except OSError:
            return

    def _rotate(self) -> None:
        try:
            self._fp.close()
        except OSError:
            pass
        bak = self.path.with_suffix(self.path.suffix + ".1")
        try:
            if bak.is_file():
                bak.unlink()
            if self.path.is_file():
                self.path.unlink()
        except OSError:
            pass
        self._fp = self.path.open("w", encoding="utf-8")
        self._fp.write(f"[rotated] kind={self.kind} cap_bytes={self.cap}\n")

    def close(self) -> None:
        try:
            self._fp.flush()
            self._fp.close()
        except OSError:
            pass

    def __enter__(self) -> "CappedLog":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
