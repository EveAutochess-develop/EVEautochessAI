"""Persistent seat Prepare: shop loop from genome weights, not ranking table."""

from __future__ import annotations

import random
from typing import Any

from eveac_ai.economy import xp_demand
from eveac_ai.prepare import (
    is_cyno_flagship,
    is_covert_cyno,
    is_shop_combat_hull,
    legal_field_cells,
    _place_relative,
)
from eveac_ai.shop import roll_equip_shop, roll_ship_shop, synth_other_halves

MAX_STEPS = 24


def _w_ship(genome: dict, titan: str, sid: str) -> float:
    w = float((((genome.get("titan_slices") or {}).get(titan) or {}).get("ship") or {}).get(str(sid), 0.4))
    return w


def _w_eq(genome: dict, titan: str, eid: str) -> float:
    return float((((genome.get("titan_slices") or {}).get(titan) or {}).get("equip") or {}).get(str(eid), 0.4))


def new_seat_board() -> dict[str, Any]:
    return {
        "pieces": [],
        "bag": [],
        "level": 1,
        "xp": 0,
        "gold": 5,
        "win_streak": 0,
        "loss_streak": 0,
        "shop_ships": [],
        "shop_equips": [],
        "token": 1,
    }


def try_merge(pieces: list[dict[str, Any]]) -> None:
    by: dict[tuple[str, int], list[dict]] = {}
    for p in pieces:
        by.setdefault((str(p["ship_id"]), int(p.get("star") or 1)), []).append(p)
    drop: set[int] = set()
    for (sid, star), grp in list(by.items()):
        if star >= 3 or len(grp) < 3:
            continue
        keep, a, b = grp[0], grp[1], grp[2]
        keep["star"] = star + 1
        keep["equips"] = []
        drop.add(id(a))
        drop.add(id(b))
    pieces[:] = [p for p in pieces if id(p) not in drop]


def _pop_cap(board: dict) -> int:
    return int(board.get("level") or 1)


def deploy_field(content: Any, genome: dict, board_state: dict, board_desc: dict) -> None:
    pieces = board_state["pieces"]
    cap = _pop_cap(board_state)
    field = [p for p in pieces if p["slot"] == "field"]
    hangar = [p for p in pieces if p["slot"] == "hangar"]
    for p in field:
        if is_cyno_flagship(content.ships.get(str(p["ship_id"]))):
            p["slot"] = "hangar"
            hangar.append(p)
    field = [p for p in pieces if p["slot"] == "field"]
    while len(field) < cap:
        cand = [p for p in hangar if not is_cyno_flagship(content.ships.get(str(p["ship_id"])))]
        if not cand:
            break
        p = cand[0]
        p["slot"] = "field"
        hangar.remove(p)
        field.append(p)
    while len(field) > cap:
        p = field[-1]
        p["slot"] = "hangar"
        field.remove(p)
    stance = genome.get("stance") or {}
    fps = [p for p in pieces if p["slot"] == "field"]
    laid = _place_relative([p["ship_id"] for p in fps], content, board_desc, stance)
    used: set[tuple[int, int]] = set()
    by = {}
    for sid, x, z in laid:
        by.setdefault(sid, []).append((x, z))
    for p in fps:
        opts = by.get(p["ship_id"]) or []
        xz = None
        while opts:
            xz = opts.pop(0)
            if xz not in used:
                break
            xz = None
        if xz is None:
            cells = [c for c in legal_field_cells(board_desc) if c not in used]
            xz = cells[0] if cells else (0, 0)
        p["x"], p["z"] = xz
        used.add(xz)


def prepare_turn(
    content: Any,
    genome: dict,
    titan: str,
    board_state: dict,
    rng: random.Random,
    board_desc: dict,
    nets=None,
    *,
    seat_id: int = 0,
    rnd: int = 0,
    titan_hp: float = 100.0,
    seats: list | None = None,
    round_kind: str = "pvp",
    security_mode: str = "nullsec",
    scripted: str | None = None,
) -> dict | None:
    if scripted:
        from eveac_ai.scripted_fsm import prepare_miner_flag

        return prepare_miner_flag(content, genome, titan, board_state, rng, board_desc)
    if nets is not None:
        return nets.prepare_seat(
            content,
            genome,
            titan,
            board_state,
            rng,
            board_desc,
            seat_id=seat_id,
            rnd=rnd,
            titan_hp=titan_hp,
            seats=seats,
            round_kind=round_kind,
            security_mode=security_mode,
        )
    econ = content.economy
    board_state["shop_ships"] = roll_ship_shop(content, level=board_state["level"], titan=titan, rng=rng)
    board_state["shop_equips"] = roll_equip_shop(content, level=board_state["level"], rng=rng)
    steps = 0
    while steps < MAX_STEPS:
        steps += 1
        gold = int(board_state["gold"])
        field_n = sum(1 for p in board_state["pieces"] if p["slot"] == "field")
        if field_n >= _pop_cap(board_state) and gold >= int(econ.get("buy_exp_gold_cost") or 4):
            if board_state["level"] < int(econ.get("player_level_cap") or 20):
                board_state["gold"] -= int(econ.get("buy_exp_gold_cost") or 4)
                board_state["xp"] += int(econ.get("buy_exp_amount") or 4)
                need = xp_demand(econ, board_state["level"])
                if board_state["xp"] >= need:
                    board_state["xp"] -= need
                    board_state["level"] += 1
                continue
        best_i, best_w = -1, -1.0
        for i, sid in enumerate(board_state["shop_ships"]):
            hull = content.ships.get(str(sid)) or {}
            cost = int(float(hull.get("cost") or 99))
            if cost > gold or not is_shop_combat_hull(hull):
                continue
            w = _w_ship(genome, titan, sid)
            if w > best_w:
                best_w, best_i = w, i
        if best_i >= 0:
            sid = board_state["shop_ships"][best_i]
            hull = content.ships.get(str(sid)) or {}
            cost = int(float(hull.get("cost") or 0))
            board_state["gold"] -= cost
            slot = "hangar"
            tok = int(board_state["token"])
            board_state["token"] = tok + 1
            board_state["pieces"].append(
                {"token": tok, "ship_id": str(sid), "star": 1, "equips": [], "slot": slot, "x": 0, "z": 0}
            )
            board_state["shop_ships"][best_i] = ""
            try_merge(board_state["pieces"])
            continue
        best_e, best_ew = "", -1.0
        for eid in board_state["shop_equips"]:
            if not eid:
                continue
            meta = content.equip_meta.get(eid) or {}
            cost = int(float(meta.get("cost") or 99))
            if cost > gold:
                continue
            w = _w_eq(genome, titan, eid)
            if w > best_ew:
                best_ew, best_e = w, eid
        if best_e:
            meta = content.equip_meta.get(best_e) or {}
            board_state["gold"] -= int(float(meta.get("cost") or 0))
            board_state["bag"].append(best_e)
            board_state["shop_equips"] = [e if e != best_e else "" for e in board_state["shop_equips"]]
            continue
        if gold >= int(econ.get("refresh_cost") or 2):
            board_state["gold"] -= int(econ.get("refresh_cost") or 2)
            board_state["shop_ships"] = roll_ship_shop(content, level=board_state["level"], titan=titan, rng=rng)
            board_state["shop_equips"] = roll_equip_shop(content, level=board_state["level"], rng=rng)
            continue
        if gold >= int(econ.get("ship_scanner_cost") or 50):
            halves = synth_other_halves(content, board_state["bag"] + [str(e).split(":")[0] for p in board_state["pieces"] for e in (p.get("equips") or [])])
            board_state["gold"] -= int(econ.get("ship_scanner_cost") or 50)
            owned = [p["ship_id"] for p in board_state["pieces"]]
            board_state["shop_ships"] = roll_ship_shop(
                content, level=board_state["level"], titan=titan, rng=rng, owned_ids=owned, scanner=not halves
            )
            board_state["shop_equips"] = roll_equip_shop(content, level=board_state["level"], rng=rng, synth_halves=halves or None)
            continue
        break
    for eid in list(board_state["bag"]):
        meta = content.equip_meta.get(eid) or {}
        placed = False
        for p in board_state["pieces"]:
            if is_covert_cyno(content.ships.get(str(p["ship_id"]))):
                continue
            if len(p.get("equips") or []) >= 3:
                continue
            p.setdefault("equips", []).append(f"{eid}:{meta.get('name') or eid}")
            board_state["bag"].remove(eid)
            placed = True
            break
        if not placed:
            break
    deploy_field(content, genome, board_state, board_desc)
