#!/usr/bin/env python3
"""Ranking stability scaffold: multi-seed jitter of genome → T0–T48 name churn."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _tier_map(table: dict) -> dict[str, str]:
    out: dict[str, str] = {}
    for tname, blob in (table.get("tiers") or {}).items():
        for it in blob.get("items") or []:
            out[str(it.get("id") or it.get("key") or "")] = str(tname)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("nullsec", "lowsec", "any"), default="any")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    samples = ROOT / "samples"
    candidates = []
    if args.mode in ("nullsec", "any"):
        p = samples / "weights_table_nullsec.json"
        if p.is_file():
            candidates.append(p)
    if args.mode in ("lowsec", "any"):
        p = samples / "weights_table_lowsec.json"
        if p.is_file():
            candidates.append(p)
    if not candidates:
        p = samples / "weights_table.json"
        if p.is_file():
            candidates.append(p)
    lines = [f"ranking_stability mode={args.mode} files={len(candidates)}"]
    for path in candidates:
        table = json.loads(path.read_text(encoding="utf-8"))
        base = _tier_map(table)
        flips = 0
        rng = random.Random(1)
        keys = [k for k in base if k]
        for s in range(args.seeds):
            # Placeholder: permute unknown tier membership noise for scaffold.
            noisy = dict(base)
            for k in keys:
                if rng.random() < 0.02:
                    noisy[k] = "TU"
                    flips += 1
            _ = s
        lines.append(f"{path.name} items={len(base)} synthetic_flips={flips} (scaffold)")
    out = samples / "ranking_stability.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
