"""Shop reveal: 6 ships + 4 equips. No seed peeking; only revealed slots + pool odds."""

from __future__ import annotations

import random
from typing import Any

from eveac_ai.prepare import is_cyno_flagship, is_covert_cyno, is_shop_combat_hull
from eveac_ai.ranking import is_gold_equip, is_titan_hull

TIER_COSTS = [2, 3, 5, 7, 13]
SIZE_LVL = {"S": 1, "M": 5, "L": 10, "XL": 15}
GROUP_UNLOCK = {
    "frigate": 1,
    "destroyer": 2,
    "cruiser": 5,
    "battlecruiser": 7,
    "battleship": 13,
    "carrier": 15,
    "dreadnought": 15,
    "force_auxiliary": 15,
    "mining_barge": 2,
    "industrial_command": 8,
    "capital_industrial": 15,
}


def _unlock_map(econ: dict[str, Any]) -> dict[str, int]:
    raw = econ.get("shop_unlock_level_by_group") or {}
    out = dict(GROUP_UNLOCK)
    for k, v in raw.items():
        out[str(k)] = int(v)
    return out


def ship_unlocked(hull: dict[str, Any], level: int, econ: dict[str, Any]) -> bool:
    if not is_shop_combat_hull(hull) or is_titan_hull(hull):
        return False
    um = _unlock_map(econ)
    g = str(hull.get("ship_group") or "frigate").lower()
    need = int(hull.get("shop_min_level") or um.get(g, 1))
    need = max(need, um.get(g, 1))
    return int(level) >= need


def equip_unlocked(meta: dict[str, Any], level: int) -> bool:
    if not is_gold_equip(meta):
        return False
    if meta.get("implant"):
        return False
    sz = str(meta.get("size") or "S").upper()
    return int(level) >= SIZE_LVL.get(sz, 1)


def _odds_row(econ: dict[str, Any], level: int) -> list[float]:
    rows = econ.get("shop_odds_by_level") or []
    idx = min(max(int(level), 1), 5) - 1
    if idx < len(rows):
        return [float(x) for x in rows[idx]]
    return [100.0, 0, 0, 0, 0]


def in_capital_pool(hull: dict[str, Any], econ: dict[str, Any] | None = None) -> bool:
    """Godot shop_controller capital roll: high cost ∪ cyno ∪ cyno-flagship ∪ rorqual."""
    econ = econ or {}
    costs = {int(x) for x in (econ.get("shop_capital_costs") or [22, 24, 37])}
    costs.add(37)
    c = int(float(hull.get("cost") or 0))
    g = str(hull.get("ship_group") or "").lower()
    role = str(hull.get("capital_role") or "").lower()
    return (
        c in costs
        or is_covert_cyno(hull)
        or is_cyno_flagship(hull)
        or g == "capital_industrial"
        or role == "capital_industrial"
    )


def _weighted_pick(rng: random.Random, cand: list[str], weights: list[float], slots: list[str]) -> str:
    tot = sum(weights) or 1.0
    x = rng.random() * tot
    acc = 0.0
    pick = cand[-1]
    for sid, w in zip(cand, weights):
        acc += w
        if x <= acc:
            pick = sid
            break
    if slots.count(pick) >= 2:
        alt = [s for s in cand if slots.count(s) < 2]
        pick = rng.choice(alt) if alt else pick
    return pick


def _pool_weights(content: Any, cand: list[str], race: str, mul: float) -> list[float]:
    out: list[float] = []
    for sid in cand:
        h = content.ships[sid]
        w = 1.0
        if str(h.get("race") or "").lower() == race:
            w *= mul
        out.append(w)
    return out


def _pick_tier_cost(rng: random.Random, odds: list[float]) -> int:
    s = sum(odds) or 1.0
    x = rng.random() * s
    acc = 0.0
    for c, w in zip(TIER_COSTS, odds):
        acc += w
        if x <= acc:
            return int(c)
    return TIER_COSTS[0]


def roll_ship_shop(
    content: Any,
    *,
    level: int,
    titan: str,
    rng: random.Random,
    owned_ids: list[str] | None = None,
    scanner: bool = False,
) -> list[str]:
    econ = content.economy
    n = int(econ.get("shop_slot_count") or 6)
    if scanner and owned_ids:
        pool = [str(s) for s in owned_ids if ship_unlocked(content.ships.get(str(s)) or {}, level, econ)]
        if pool:
            return [rng.choice(pool) for _ in range(n)]
    odds = _odds_row(econ, level)
    race = titan.replace("titan_", "")
    mul = float(econ.get("titan_shop_race_weight_mul") or 1.1)
    cap_lv = int(econ.get("shop_capital_min_level") or 15)
    cap_w = float(econ.get("shop_capital_roll_weight_pct") or 12)
    slots: list[str] = []
    for _ in range(n):
        if int(level) >= cap_lv and cap_w > 0 and rng.random() * 100.0 < cap_w:
            cap_cand = [
                sid
                for sid, h in content.ships.items()
                if ship_unlocked(h, level, econ) and in_capital_pool(h, econ)
            ]
            if cap_cand:
                slots.append(_weighted_pick(rng, cap_cand, _pool_weights(content, cap_cand, race, mul), slots))
                continue
        cost = _pick_tier_cost(rng, odds)
        cand = [
            sid
            for sid, h in content.ships.items()
            if ship_unlocked(h, level, econ) and int(float(h.get("cost") or 0)) in (cost, cost + 1 if cost == 7 else cost)
        ]
        if cost == 7:
            cand = [sid for sid, h in content.ships.items() if ship_unlocked(h, level, econ) and int(float(h.get("cost") or 0)) in (7, 8)]
        if not cand:
            cand = [sid for sid, h in content.ships.items() if ship_unlocked(h, level, econ)]
        if not cand:
            slots.append("10")
            continue
        slots.append(_weighted_pick(rng, cand, _pool_weights(content, cand, race, mul), slots))
    return slots


def roll_equip_shop(
    content: Any,
    *,
    level: int,
    rng: random.Random,
    synth_halves: list[str] | None = None,
) -> list[str]:
    n = int(content.economy.get("equipment_shop_slot_count") or 4)
    if synth_halves:
        pool = [e for e in synth_halves if e in content.equip_meta]
        if pool:
            return [rng.choice(pool) for _ in range(n)]
    cand = [eid for eid, m in content.equip_meta.items() if equip_unlocked(m, level)]
    if not cand:
        return []
    return [rng.choice(cand) for _ in range(n)]


def synth_other_halves(content: Any, owned_eids: list[str]) -> list[str]:
    out: list[str] = []
    for eid in owned_eids:
        meta = content.equip_meta.get(str(eid).split(":", 1)[0]) or {}
        nxt = meta.get("synth_from")
        if isinstance(nxt, list) and len(nxt) == 2:
            a, b = str(nxt[0]), str(nxt[1])
            me = str(meta.get("id") or eid)
            other = b if a == me or a in me else a
            if other in content.equip_meta:
                out.append(other)
        sn = meta.get("synth_next")
        if isinstance(sn, str) and sn in content.equip_meta:
            out.append(sn)
    return out
