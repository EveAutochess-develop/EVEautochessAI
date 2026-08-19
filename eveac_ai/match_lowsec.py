"""Lowsec 1v1 self-play entry (delegates to match20 with security_mode=lowsec)."""

from __future__ import annotations

import sys

from eveac_ai.content import load_config
from eveac_ai.match20 import main as match20_main


def main(minutes: float | None = None, *, fresh: bool = False) -> None:
    cfg_path_hint = "config.json forces lowsec for this process via env override in argv helper"
    # Force lowsec for this process by patching loaded config on next main — set via argv flag.
    # match20 reads config each start; we inject by writing temporary overlay is avoided.
    # Instead: set environment-style by monkeypatching load before import of loop.
    import eveac_ai.match20 as m

    _orig = m.load_config

    def _lowsec_cfg():
        cfg = _orig()
        cfg["security_modes_enabled"] = ["lowsec"]
        cfg["lowsec_frac"] = 1.0
        cfg["_security_mode"] = "lowsec"
        return cfg

    m.load_config = _lowsec_cfg  # type: ignore[assignment]
    try:
        match20_main(minutes=minutes, fresh=fresh)
    finally:
        m.load_config = _orig  # type: ignore[assignment]


if __name__ == "__main__":
    mins = None
    fresh = False
    for a in sys.argv[1:]:
        if a == "--fresh":
            fresh = True
        elif a.startswith("--minutes="):
            mins = float(a.split("=", 1)[1])
    main(mins, fresh=fresh)
