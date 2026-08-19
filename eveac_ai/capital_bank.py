"""Fuse late StateBank tables into legal capital-bank situations. Train-only."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from eveac_ai.content import titan_pvp_loss
from eveac_ai.prepare import is_cyno_flagship, is_covert_cyno
from eveac_ai.shop import roll_ship_shop
from eveac_ai.state_bank import mode_dir, validate_league

VARIANTS = ("shop_offer", "flag_no_cyno", "cyno_no_flag")


def _pool_ids(content: Any, pred) -> list[str]:
    out = []
    for sid, h in content.ships.items():
        if pred(h):
            out.append(str(sid))
    return out


def cyno_ids(content: Any) -> list[str]:
    return _pool_ids(content, is_covert_cyno)


def flag_ids(content: Any) -> list[str]:
    return [sid for sid, h in content.ships.items() if is_cyno_flagship(h) and int(float(h.get("cost") or 0)) <= 24]


def pick_late_league(samples: Path, mode: str, rng: random.Random) -> dict[str, Any] | None:
    bucket = "r10_plus" if str(mode) == "lowsec" else "r12_plus"
    d = mode_dir(samples, mode) / bucket
    if not d.is_dir():
        return None
    files = list(d.glob("*.json"))
    if not files:
        return None
    path = rng.choice(files)
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return row.get("league") if isinstance(row, dict) else None


def _piece(sid: str, slot: str, token: int, x: int = 0, z: int = 0) -> dict[str, Any]:
    return {"token": token, "ship_id": str(sid), "star": 1, "equips": [], "slot": slot, "x": x, "z": z}


def fuse_seat(
    content: Any,
    seat: dict[str, Any],
    rng: random.Random,
    *,
    variant: str,
    loss: float,
) -> None:
    board = seat.setdefault("board", {})
    board["level"] = int(rng.randint(12, 16))
    board["xp"] = int(rng.randint(0, 20))
    board["gold"] = int(rng.randint(30, 100))
    k = int(rng.choice([1, 2, 3]))
    seat["titan_hp"] = float(max(1.0, loss) * k)
    pieces = list(board.get("pieces") or [])
    for p in pieces:
        hull = content.ships.get(str(p.get("ship_id"))) or {}
        if is_cyno_flagship(hull) and p.get("slot") == "field":
            p["slot"] = "hangar"
    tok = max([int(p.get("token") or 0) for p in pieces] + [board.get("token") or 1]) + 1
    cynos = cyno_ids(content)
    flags = flag_ids(content)
    cyno = rng.choice(cynos) if cynos else "101"
    flag = rng.choice(flags) if flags else "111"
    titan = str(seat.get("titan") or "amarr")
    shop = roll_ship_shop(content, level=int(board["level"]), titan=titan, rng=rng)
    if variant == "shop_offer":
        offer = rng.choice([cyno, flag])
        if offer not in shop:
            shop[0] = offer
        if is_cyno_flagship(content.ships.get(flag) or {}) and not any(
            is_cyno_flagship(content.ships.get(str(p.get("ship_id"))) or {}) for p in pieces
        ):
            pieces.append(_piece(flag, "hangar", tok))
            tok += 1
    elif variant == "flag_no_cyno":
        pieces = [p for p in pieces if not is_covert_cyno(content.ships.get(str(p.get("ship_id"))) or {})]
        if not any(is_cyno_flagship(content.ships.get(str(p.get("ship_id"))) or {}) for p in pieces):
            pieces.append(_piece(flag, "hangar", tok))
            tok += 1
        shop = [s for s in shop if not is_covert_cyno(content.ships.get(str(s)) or {})] or shop
        if flag not in shop:
            shop[0] = flag
    else:
        pieces = [p for p in pieces if not is_cyno_flagship(content.ships.get(str(p.get("ship_id"))) or {})]
        if not any(is_covert_cyno(content.ships.get(str(p.get("ship_id"))) or {}) for p in pieces):
            pieces.append(_piece(cyno, "hangar", tok))
            tok += 1
        if cyno not in shop:
            shop[0] = cyno
    board["pieces"] = pieces
    board["shop_ships"] = shop[:6]
    board["token"] = tok
    cap = max(1, int(board["level"]))
    field = [p for p in pieces if p.get("slot") == "field"]
    hangar = [p for p in pieces if p.get("slot") == "hangar"]
    while len(field) > cap:
        p = field.pop()
        p["slot"] = "hangar"
        hangar.append(p)


def fuse_league(content: Any, league: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    loss = titan_pvp_loss(getattr(content, "titan_pvp", None))
    for i, seat in enumerate(league.get("seats") or []):
        fuse_seat(content, seat, rng, variant=VARIANTS[i % len(VARIANTS)], loss=loss)
    return league


def apply_fuse_to_match_league(content: Any, match_league: Any, samples: Path, rng: random.Random, mode: str) -> bool:
    """Mutate MatchLeague seats from a late bank table, or fuse in place if bank empty."""
    blob = pick_late_league(samples, mode, rng)
    src_seats = (blob or {}).get("seats") if blob else None
    if src_seats and len(src_seats) == len(match_league.seats):
        for dst, src in zip(match_league.seats, src_seats):
            if dst.get("scripted"):
                continue
            if isinstance(src.get("board"), dict):
                dst["board"] = json.loads(json.dumps(src["board"]))
            if src.get("titan"):
                dst["titan"] = src["titan"]
    live = [s for s in match_league.seats if not s.get("scripted")]
    fuse_league(content, {"seats": live}, rng)
    errs = validate_league({"seats": match_league.seats}, mode=mode, content=content)
    match_league.train_source = "capital_bank"
    match_league.force_pvp = True
    return not errs
