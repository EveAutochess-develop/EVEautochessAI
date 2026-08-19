"""Whole-table StateBank: ingest mid-match checkpoints; mode-separated buckets.

Hard rules (this slice): same-mode only; whole-table reuse only (no seat stitch);
shop re-roll later; memory keep legal summaries only. Mid-resume training is next slice.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


def bank_root(samples: Path) -> Path:
    d = samples / "state_bank"
    d.mkdir(parents=True, exist_ok=True)
    return d


def mode_dir(samples: Path, mode: str) -> Path:
    m = "lowsec" if str(mode) == "lowsec" else "nullsec"
    d = bank_root(samples) / m
    d.mkdir(parents=True, exist_ok=True)
    return d


def bucket_for_round(mode: str, round_1based: int) -> str:
    r = int(round_1based)
    if str(mode) == "lowsec":
        if r <= 3:
            return "r03"
        if r <= 6:
            return "r06"
        if r <= 10:
            return "r10"
        return "r10_plus"
    if r <= 4:
        return "r04"
    if r <= 6:
        return "r06"
    if r <= 8:
        return "r08"
    if r <= 12:
        return "r05_r12"
    return "r12_plus"


def validate_league(league: dict[str, Any], *, mode: str, content: Any | None = None) -> list[str]:
    """Return list of error strings; empty means ok."""
    errs: list[str] = []
    seats = league.get("seats") or []
    n = len(seats)
    if str(mode) == "lowsec" and n != 2:
        errs.append(f"lowsec seats={n} want 2")
    if str(mode) == "nullsec" and n not in (0, 20) and n < 2:
        errs.append(f"nullsec seats={n} unexpected")
    for s in seats:
        board = s.get("board") or {}
        level = max(1, int(board.get("level") or 1))
        field = [p for p in (board.get("pieces") or []) if p.get("slot") == "field"]
        if len(field) > level + 8:
            errs.append(f"seat {s.get('seat_id')} field>{level}+8")
        for p in board.get("pieces") or []:
            star = int(p.get("star") or 1)
            if star < 1 or star > 3:
                errs.append(f"bad star {star}")
            eqs = p.get("equips") or []
            if len(eqs) > 3:
                errs.append("equips>3")
            if content is not None:
                hull = content.ships.get(str(p.get("ship_id"))) or {}
                ship_sz = str(hull.get("size") or "S")
                for raw in eqs:
                    eid = str(raw).split(":", 1)[0]
                    meta = content.equip_meta.get(eid) or {}
                    eq_sz = str(meta.get("size") or "S")
                    order = {"S": 0, "M": 1, "L": 2, "XL": 3}
                    if order.get(eq_sz, 0) > order.get(ship_sz, 3):
                        errs.append(f"equip size {eid}")
        gold = int(board.get("gold") or 0)
        if gold < 0 or gold > 500:
            errs.append(f"gold={gold}")
        hp = float(s.get("titan_hp") or 0)
        if hp < 0 or hp > 500:
            errs.append(f"titan_hp={hp}")
    return errs


def tag_value(league: dict[str, Any], *, mode: str, round_1based: int) -> list[str]:
    tags: list[str] = [f"mode:{mode}", f"r:{round_1based}"]
    for s in league.get("seats") or []:
        board = s.get("board") or {}
        gold = int(board.get("gold") or 0)
        if gold in (9, 10, 19, 20, 29, 30):
            tags.append("interest_crit")
        lives_hint = float(s.get("titan_hp") or 0)
        if 0 < lives_hint <= 60:
            tags.append("life_crit")
        field = [p for p in (board.get("pieces") or []) if p.get("slot") == "field"]
        level = int(board.get("level") or 1)
        if field and len(field) >= max(1, level - 1):
            tags.append("pop_crit")
        if board.get("bag"):
            tags.append("bag_equip")
    return sorted(set(tags))


def ingest_checkpoint(
    samples: Path,
    blob: dict[str, Any],
    *,
    mode: str,
    content: Any | None = None,
) -> int:
    """Ingest whole leagues from a match_checkpoint-like blob. Returns rows written."""
    mode = "lowsec" if str(mode) == "lowsec" else "nullsec"
    round_done = int(blob.get("round_done") or 0)
    round_1based = round_done + 1
    bucket = bucket_for_round(mode, round_1based)
    bdir = mode_dir(samples, mode) / bucket
    bdir.mkdir(parents=True, exist_ok=True)
    index = bank_root(samples) / "index.jsonl"
    n = 0
    gen_i = int(blob.get("gen_i") or 0)
    for league in blob.get("leagues") or []:
        errs = validate_league(league, mode=mode, content=content)
        if errs:
            continue
        tags = tag_value(league, mode=mode, round_1based=round_1based)
        name = f"g{gen_i}_r{round_1based}_L{league.get('league_i', 0)}.json"
        path = bdir / name
        row = {
            "schema_ver": "1",
            "security_mode": mode,
            "bucket": bucket,
            "gen_i": gen_i,
            "round_1based": round_1based,
            "tags": tags,
            "league": league,
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
        with index.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "path": str(path.relative_to(samples)).replace("\\", "/"),
                        "security_mode": mode,
                        "bucket": bucket,
                        "gen_i": gen_i,
                        "round_1based": round_1based,
                        "tags": tags,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        n += 1
    return n
