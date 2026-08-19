"""Featured fight feedback: panel residuals, signed battle grade, unit shares."""

from __future__ import annotations

import math
from typing import Any

from eveac_ai.content import star_at
from eveac_ai.prepare import is_covert_cyno, is_cyno_flagship, is_shop_combat_hull
from eveac_ai.ship import damage_sum

AXES = ("dmg", "tank", "repair", "cap")


def _f(v: Any, d: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def _log_ratio(actual: float, expected: float) -> float:
    return math.log1p(max(0.0, actual) / max(1.0, expected))


def expected_axes(content: Any, ship_id: str, star: int, battle_s: float) -> dict[str, float]:
    hull = content.ships.get(str(ship_id)) or {}
    st = star_at(hull, max(1, int(star)))
    dmg = st.get("damage") if isinstance(st.get("damage"), dict) else {}
    dsum = damage_sum({k: _f(dmg.get(k)) for k in ("emp", "thermal", "kinetic", "explosive")})
    cycle = max(0.4, _f(hull.get("attack_cycle_s"), 2.0))
    shots = max(1.0, battle_s / cycle)
    ehp = _f(st.get("shield_hp")) + _f(st.get("armor_hp")) + _f(st.get("structure_hp"))
    logi = bool(hull.get("is_logistic") or st.get("is_logistic"))
    cap = _f(hull.get("capacitor_capacity"))
    return {
        "dmg": max(1.0, dsum * shots * 0.35),
        "tank": max(1.0, ehp),
        "repair": max(1.0, ehp * 0.4) if logi else 1.0,
        "cap": max(1.0, cap * 0.25) if dsum > 1.0 else 1.0,
    }


def residuals_for_ship(content: Any, row: dict[str, Any], battle_s: float) -> dict[str, float]:
    sid = str(row.get("ship_id") or "")
    exp = expected_axes(content, sid, int(row.get("star") or 1), battle_s)
    actual = {
        "dmg": _f(row.get("dmg_out")),
        "tank": _f(row.get("dmg_in")) + (80.0 if row.get("survived") else 0.0),
        "repair": _f(row.get("repaired")),
        "cap": _f(row.get("cap_used")),
    }
    return {k: _log_ratio(actual[k], exp[k]) for k in AXES}


def first_kill_s(kills: list, ships: list[dict]) -> float:
    uids = {str(s.get("uid")) for s in ships}
    ts = [float(k.get("t") or 0) for k in kills if str(k.get("killer")) in uids or str(k.get("killer_uid")) in uids]
    if not ts:
        for k in kills:
            if k.get("killer_ship_id"):
                ts.append(float(k.get("t") or 0))
    return min(ts) if ts else 900.0


def eval_valence_delta(content: Any, row: dict, pieces: list[dict], won: bool) -> float:
    """Signed title conditions. Count of titles is never a reward."""
    ships = [s for s in (row.get("ships") or []) if not s.get("is_unmanned")]
    dmg = [max(0.0, _f(s.get("dmg_out"))) for s in ships]
    tot = sum(dmg) or 1.0
    acc = 0.0
    if ships and max(dmg) / tot >= 0.40:
        acc += 0.08  # efficient_firepower +
    cyno_field = False
    flag_hangar = False
    for p in pieces or []:
        hull = content.ships.get(str(p.get("ship_id"))) or {}
        if p.get("slot") == "field" and is_covert_cyno(hull):
            cyno_field = True
        if p.get("slot") == "hangar" and is_cyno_flagship(hull):
            flag_hangar = True
    # waiting_for_godot: cyno on field with nothing to jump, then lose.
    if cyno_field and not flag_hangar and not won:
        acc -= 0.12
    acc += cyno_key_delta(content, pieces or [])
    hulls = [content.ships.get(str(p.get("ship_id"))) or {} for p in (pieces or []) if p.get("slot") == "field"]
    if hulls and all(h.get("is_mining_ship") for h in hulls) and won:
        acc += 0.05
    return acc


def cyno_key_delta(content: Any, pieces: list[dict], *, lam: float = 0.08) -> float:
    """Hangar flagship → cyno on field is the jump key. Smaller than pop/XP credit."""
    flag_hangar = False
    cyno_field = False
    cyno_hangar = False
    for p in pieces or []:
        hull = content.ships.get(str(p.get("ship_id"))) or {}
        if is_cyno_flagship(hull) and p.get("slot") == "hangar":
            flag_hangar = True
        if is_covert_cyno(hull):
            if p.get("slot") == "field":
                cyno_field = True
            elif p.get("slot") == "hangar":
                cyno_hangar = True
    if not flag_hangar:
        return 0.0
    if cyno_field:
        return float(lam)
    if cyno_hangar:
        return -float(lam)
    return -0.5 * float(lam)


def path_value(
    *,
    level: int,
    field_n: int,
    lives: int,
    max_lives: int,
    med_level: float,
    med_pop: float,
) -> float:
    """Remaining-match value: pop/level vs table median + lives. Not 'has capital'."""
    lv = (float(level) - float(med_level)) / 20.0
    pop = (float(field_n) - float(med_pop)) / 10.0
    life = (float(lives) / max(1.0, float(max_lives))) - 0.5
    return math.tanh(1.2 * lv + 0.9 * pop + 0.5 * life)


def xp_shape_delta(field_full: bool, bought_xp: bool, *, lam: float = 0.12) -> float:
    """Bootstrap only. Smaller than n-step path credit, larger than cyno key."""
    if bought_xp and field_full:
        return float(lam)
    if bought_xp and not field_full:
        return -0.4 * float(lam)
    if (not bought_xp) and field_full:
        return -0.5 * float(lam)
    return 0.0


def table_grade(
    collab: dict,
    *,
    won: bool,
    draw: bool,
    dmg_self: float,
    dmg_enemy: float,
    gold_self: float,
    gold_enemy: float,
    d_hp: float,
    pop_self: float,
    pop_enemy: float,
    first_kill: float,
    wipe: bool,
    eval_delta: float,
) -> float:
    """Signed table grade. d_hp is -1 when this PVP round lost a life, else 0."""
    edge = (dmg_self - dmg_enemy) / max(1.0, dmg_self + dmg_enemy)
    gold_pos = (gold_self - gold_enemy) / 50.0
    pop = (pop_self - pop_enemy) / 30.0
    fk = max(0.0, 1.0 - first_kill / 120.0)
    raw = (
        edge
        + float(collab.get("lam_econ") or 1.2) * gold_pos
        + float(collab.get("lam_hp") or 0.4) * d_hp
        + float(collab.get("lam_pop") or 0.2) * pop
        + float(collab.get("lam_first_kill") or 0.12) * fk
        + (0.08 if wipe else 0.0)
        + eval_delta
    )
    if draw:
        raw -= 0.05
    elif won:
        raw += 0.15
    else:
        raw -= 0.15
    return math.tanh(raw)


def unit_credit(content: Any, row: dict, battle_s: float, collab: dict) -> dict[str, dict[str, float]]:
    """ship_id -> axes + share c + kill weight. Unmanned / non-shop skipped."""
    kills = row.get("kill_calendar") or []
    kill_by: dict[str, float] = {}
    uid_to_sid = {str(s.get("uid")): str(s.get("ship_id") or "") for s in row.get("ships") or []}
    for k in kills:
        sid = str(k.get("killer_ship_id") or "")
        if not sid:
            sid = uid_to_sid.get(str(k.get("killer") or ""), "")
        if not sid:
            continue
        w = 0.35 if k.get("victim_unmanned") else 1.0
        t = float(k.get("t") or 0.0)
        w *= max(0.25, 1.0 - t / max(60.0, battle_s))
        kill_by[sid] = kill_by.get(sid, 0.0) + w
    out: dict[str, dict[str, float]] = {}
    weights: dict[str, float] = {}
    for s in row.get("ships") or []:
        if s.get("is_unmanned"):
            continue
        sid = str(s.get("ship_id") or "")
        hull = content.ships.get(sid) or {}
        if not is_shop_combat_hull(hull):
            continue
        res = residuals_for_ship(content, s, battle_s)
        live = 1.0 if s.get("survived") else 0.0
        kk = kill_by.get(sid, 0.0)
        w = max(
            0.0,
            float(collab.get("lam_dmg") or 1.0) * res["dmg"]
            + float(collab.get("lam_tank") or 0.7) * res["tank"]
            + float(collab.get("lam_repair") or 0.8) * res["repair"]
            + float(collab.get("lam_cap") or 0.4) * res["cap"]
            + float(collab.get("lam_kill") or 0.5) * kk
            + float(collab.get("lam_live") or 0.3) * live,
        )
        weights[sid] = weights.get(sid, 0.0) + w
        prev = out.get(sid)
        if prev is None:
            out[sid] = {**res, "w": w, "c": 0.0, "kill": kk}
        else:
            for a in AXES:
                prev[a] = prev[a] + res[a]
            prev["w"] += w
            prev["kill"] += kk
    tot = sum(weights.values())
    n = max(1, len(out))
    for sid, rec in out.items():
        rec["c"] = (rec["w"] / tot) if tot > 1e-8 else (1.0 / n)
    return out


def equip_credit(
    content: Any,
    pieces: list[dict],
    ship_credit: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    """Split each ship's unit_credit evenly across gold equips mounted on living manned hulls."""
    from eveac_ai.ranking import is_gold_equip

    out: dict[str, dict[str, float]] = {}
    for p in pieces or []:
        if p.get("slot") != "field":
            continue
        sid = str(p.get("ship_id") or "")
        rec = ship_credit.get(sid)
        if not rec:
            continue
        hull = content.ships.get(sid) or {}
        if not is_shop_combat_hull(hull):
            continue
        eids: list[str] = []
        for raw in p.get("equips") or []:
            eid = str(raw).split(":", 1)[0]
            meta = content.equip_meta.get(eid) or {}
            if is_gold_equip(meta):
                eids.append(eid)
        if not eids:
            continue
        share = 1.0 / float(len(eids))
        for eid in eids:
            prev = out.get(eid)
            chunk = {k: float(rec.get(k, 0.0)) * share for k in (*AXES, "w", "c", "kill")}
            if prev is None:
                out[eid] = chunk
            else:
                for k, v in chunk.items():
                    prev[k] = float(prev.get(k, 0.0)) + v
    return out


def merge_ship_equip_credit(
    content: Any,
    pieces: list[dict],
    ship_credit: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    """Ship ids + equip ids in one map for ShopNet buy_delta / genome writes."""
    merged = {str(k): dict(v) for k, v in (ship_credit or {}).items()}
    for eid, rec in equip_credit(content, pieces, ship_credit).items():
        merged[str(eid)] = rec
    return merged


SOURCE_WEIGHTS = {
    "natural": 1.0,
    "nullsec": 1.0,
    "lowsec": 1.0,
    "whole_table": 0.7,
    "seat_stitch": 0.4,
    "synthetic_rare": 0.2,
    "capital_bank": 0.5,
}


def source_weight(kind: str | None) -> float:
    return float(SOURCE_WEIGHTS.get(str(kind or "natural"), 1.0))


def apply_genome_delta(
    genome: dict,
    titan: str,
    credit: dict[str, dict[str, float]],
    won: bool,
    *,
    content: Any | None = None,
    source: str = "natural",
) -> None:
    sl = (genome.get("titan_slices") or {}).get(titan) or {}
    ships = sl.setdefault("ship", {})
    equips = sl.setdefault("equip", {})
    step = (0.03 if won else 0.02) * source_weight(source)
    equip_ids = set()
    if content is not None:
        equip_ids = set(str(k) for k in (getattr(content, "equip_meta", None) or {}))
    for kid, rec in credit.items():
        key = str(kid)
        mag = max(-1.0, min(1.0, rec.get("w", 0.0) - 0.4 + rec.get("c", 0.0)))
        if key in equip_ids or (content is None and key in equips):
            old = float(equips.get(key, 0.45))
            equips[key] = round(max(0.05, min(0.99, old + step * mag)), 4)
        else:
            old = float(ships.get(key, 0.45))
            ships[key] = round(max(0.05, min(0.99, old + step * mag)), 4)


def ema_blend(store: dict[str, list[float]], credit: dict[str, dict[str, float]], alpha: float = 0.2) -> None:
    for sid, rec in credit.items():
        cur = store.get(sid) or [0.0, 0.0, 0.0, 0.0]
        nxt = [rec.get(a, 0.0) for a in AXES]
        store[str(sid)] = [((1.0 - alpha) * c + alpha * n) for c, n in zip(cur, nxt)]


def buy_delta(collab: dict, *, grade: float, rec: dict[str, float] | None, table_rest: float) -> float:
    clip = float(collab.get("clip") or 0.2) * 5
    if rec is None:
        raw = float(collab.get("lam_grade") or 0.35) * grade * 0.15 + float(collab.get("lam_table") or 0.25) * table_rest
        return max(-clip, min(clip, raw))
    raw = (
        float(collab.get("lam_grade") or 0.35) * grade * float(rec.get("c") or 0.0)
        + float(collab.get("lam_dmg") or 1.0) * 0.15 * rec.get("dmg", 0.0)
        + float(collab.get("lam_tank") or 0.7) * 0.10 * rec.get("tank", 0.0)
        + float(collab.get("lam_repair") or 0.8) * 0.10 * rec.get("repair", 0.0)
        + float(collab.get("lam_cap") or 0.4) * 0.08 * rec.get("cap", 0.0)
        + float(collab.get("lam_table") or 0.25) * table_rest
    )
    return max(-clip, min(clip, raw))
