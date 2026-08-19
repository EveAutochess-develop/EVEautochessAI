#!/usr/bin/env python3
"""Offline reward ablation scaffold (does not interrupt default training).

Example:
  python -m eveac_ai.tools.ablation_reward --batches 0 --mode nullsec
Writes a stub plan under samples/ablation/ when --batches 0 (dry doc only).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, default=0, help="0 = write recipe only; >0 reserved for short runs")
    ap.add_argument("--mode", choices=("nullsec", "lowsec"), default="nullsec")
    ap.add_argument("--parallel", type=int, default=1)
    args = ap.parse_args()
    out = ROOT / "samples" / "ablation"
    out.mkdir(parents=True, exist_ok=True)
    recipe = {
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "mode": args.mode,
        "batches": args.batches,
        "parallel": args.parallel,
        "note": "Stress-test signal only; do not auto-write Godot content.",
        "variants": [
            {"lam_econ": 0},
            {"lam_hp": 0},
            {"lam_dmg": 0},
            {"lam_live": 0},
        ],
        "ranking_is": "pressure_test_not_truth",
    }
    path = out / f"recipe_{args.mode}.json"
    path.write_text(json.dumps(recipe, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {path}")
    if args.batches > 0:
        print("short live ablation not auto-run here; use match20 with collab overrides manually")


if __name__ == "__main__":
    main()
