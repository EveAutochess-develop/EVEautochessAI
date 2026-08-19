"""Mixed lance — CAPITAL_AND_CYNO §4.1 / mixed_lance.gd."""

from __future__ import annotations

import math
from typing import Any

from eveac_ai.ship import SimShip, apply_hit

PHASE_IDLE = 0
PHASE_PREP = 1
PHASE_FIRE = 2
PHASE_END = 3
EL_LIMIT = math.pi / 6.0
MODULE_ID = "mixed_lance"


def _f(v: Any, d: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def lance_def(content: Any) -> dict[str, Any]:
    return (getattr(content, "equip_meta", None) or {}).get(MODULE_ID) or {}


def hull_can_lance(ship: SimShip) -> bool:
    return ship.capital_role == "dreadnought" or ship.ship_group == "dreadnought"


def board_diagonal_cells(content: Any) -> float:
    b = getattr(content, "board", None) or {}
    sx = _f(b.get("field_width"), 12.0) * 2.0
    sz = _f(b.get("field_height"), 6.0) * 2.0 + _f(b.get("center_gap_z"), 4.0)
    return max(1.0, math.hypot(sx, sz))


def _dir_el_clamp(dx: float, dy: float, dz: float) -> tuple[float, float, float]:
    az = math.atan2(dx, dz)
    horiz = math.hypot(dx, dz)
    el = max(-EL_LIMIT, min(EL_LIMIT, math.atan2(dy, max(horiz, 1e-4))))
    c = math.cos(el)
    return math.sin(az) * c, math.sin(el), math.cos(az) * c


def sphere_hits_cylinder(
    cx: float,
    cy: float,
    cz: float,
    sphere_r: float,
    ox: float,
    oy: float,
    oz: float,
    dx: float,
    dy: float,
    dz: float,
    height: float,
    cyl_r: float,
) -> bool:
    ln = math.sqrt(dx * dx + dy * dy + dz * dz) or 1.0
    dx, dy, dz = dx / ln, dy / ln, dz / ln
    relx, rely, relz = cx - ox, cy - oy, cz - oz
    along = relx * dx + rely * dy + relz * dz
    tt = max(0.0, min(max(0.0, height), along))
    px, py, pz = ox + dx * tt, oy + dy * tt, oz + dz * tt
    lim = max(0.0, cyl_r) + max(0.0, sphere_r)
    d2 = (cx - px) ** 2 + (cy - py) ** 2 + (cz - pz) ** 2
    return d2 <= lim * lim


def _collision_r(s: SimShip) -> float:
    return 0.18 if s.is_unmanned else 0.4


def _pick_target(s: SimShip, ships: list[SimShip]) -> SimShip | None:
    best_dread = None
    best_dread_d = 1e18
    best_any = None
    best_any_d = 1e18
    for o in ships:
        if o.uid == s.uid or o.team == s.team or not o.alive():
            continue
        d = s.dist_to(o)
        if o.capital_role == "dreadnought" or o.ship_group == "dreadnought":
            if d < best_dread_d:
                best_dread_d = d
                best_dread = o
        if d < best_any_d:
            best_any_d = d
            best_any = o
    return best_dread or best_any


def _ready(s: SimShip, t: float) -> bool:
    return s.alive() and (not s.is_unmanned) and t >= s.hold_until and s.has_lance and not s.lance_spent


def flush_salvo(ships: list[SimShip], content: Any, t: float) -> None:
    holders = [s for s in ships if s.has_lance and not s.lance_spent and s.lance_phase == PHASE_IDLE and s.alive()]
    if not holders:
        return
    if any(not _ready(s, t) for s in holders):
        return
    eligible = []
    for s in holders:
        tgt = _pick_target(s, ships)
        if tgt is None:
            continue
        eligible.append((s, tgt))
    if not eligible:
        return
    beam_h = board_diagonal_cells(content)
    for s, tgt in eligible:
        dx, dy, dz = tgt.x - s.x, tgt.y - s.y, tgt.z - s.z
        if dx * dx + dy * dy + dz * dz < 1e-8:
            dx, dy, dz = 0.0, 0.0, 1.0
        dx, dy, dz = _dir_el_clamp(dx, dy, dz)
        s.lance_phase = PHASE_PREP
        s.lance_phase_t = 0.0
        s.lance_tick_acc = 0.0
        s.lance_ox, s.lance_oy, s.lance_oz = s.x, s.y, s.z
        s.lance_dx, s.lance_dy, s.lance_dz = dx, dy, dz
        s.lance_beam_h = beam_h


def _apply_column(
    s: SimShip, ships: list[SimShip], dfn: dict[str, Any], shrink: float, pierce: bool, t: float, kills: list, revive_q: list | None
) -> None:
    wu = 3.0
    beam_h = s.lance_beam_h * shrink
    diam = _f(dfn.get("attack_diameter"), 2.5)
    radius = (diam * 0.5 / wu) * shrink
    if beam_h < 0.05 or radius < 0.01:
        return
    pct = _f(dfn.get("damage_hp_pct"), 0.05)
    floor_v = _f(dfn.get("damage_floor"), 1000.0)
    heal_mul = _f(dfn.get("heal_received_mul"), 0.2)
    heal_dur = _f(dfn.get("heal_debuff_sec"), 60.0)
    spd_mul = _f(dfn.get("speed_mul"), 0.1)
    spd_dur = _f(dfn.get("speed_debuff_sec"), 60.0)
    for o in ships:
        if not o.alive():
            continue
        if not sphere_hits_cylinder(
            o.x, o.y, o.z, _collision_r(o),
            s.lance_ox, s.lance_oy, s.lance_oz,
            s.lance_dx, s.lance_dy, s.lance_dz,
            beam_h, radius,
        ):
            continue
        max_hp = o.max_shield + o.max_armor + o.max_structure
        raw = max(floor_v, max_hp * pct)
        q = raw * 0.25
        res = apply_hit(o, {"emp": q, "thermal": q, "kinetic": q, "explosive": q}, pierce=pierce)
        s.dmg_out += float(res["dealt"])
        o.heal_recv_mul = heal_mul
        o.heal_debuff_until = t + heal_dur
        if o.speed_debuff_until <= t:
            o.speed_base = o.speed
        o.speed = o.speed_base * spd_mul
        o.speed_debuff_until = t + spd_dur
        if res["destroyed"]:
            kills.append({"t": round(t, 3), "killer": s.uid, "victim": o.uid, "ship_id": o.ship_id, "via": "mixed_lance"})
            if revive_q is not None and o.is_unmanned and o.mother_uid > 0:
                revive_q.append(
                    {"mother_uid": o.mother_uid, "drone_id": o.ship_id, "revive_at": t + 400.0, "star": max(1, int(o.star))}
                )


def tick_lances(
    ships: list[SimShip], content: Any, dt: float, t: float, pierce: bool, kills: list, revive_q: list | None = None
) -> None:
    dfn = lance_def(content)
    prep = _f(dfn.get("prep_sec"), 10.0)
    fire = _f(dfn.get("fire_sec"), 10.0)
    end_sec = max(0.05, _f(dfn.get("end_sec"), 2.1))
    tick_sec = max(0.05, _f(dfn.get("tick_sec"), 0.5))
    for s in ships:
        if t >= s.heal_debuff_until:
            s.heal_recv_mul = 1.0
        if t >= s.speed_debuff_until and s.speed_debuff_until > 0:
            s.speed = s.speed_base
            s.speed_debuff_until = 0.0
        if s.lance_phase == PHASE_IDLE:
            continue
        if not s.alive():
            s.lance_phase = PHASE_IDLE
            s.lance_spent = True
            continue
        s.lance_phase_t += dt
        if s.lance_phase == PHASE_PREP:
            if s.lance_phase_t >= prep:
                s.lance_phase = PHASE_FIRE
                s.lance_phase_t = 0.0
                s.lance_tick_acc = 0.0
        elif s.lance_phase == PHASE_FIRE:
            s.lance_tick_acc += dt
            while s.lance_tick_acc >= tick_sec:
                s.lance_tick_acc -= tick_sec
                _apply_column(s, ships, dfn, 1.0, pierce, t, kills, revive_q)
            if s.lance_phase_t >= fire:
                s.lance_phase = PHASE_END
                s.lance_phase_t = 0.0
        elif s.lance_phase == PHASE_END:
            shrink = max(0.0, min(1.0, 1.0 - s.lance_phase_t / end_sec))
            s.lance_tick_acc += dt
            while s.lance_tick_acc >= tick_sec and shrink > 0.02:
                s.lance_tick_acc -= tick_sec
                _apply_column(s, ships, dfn, shrink, pierce, t, kills, revive_q)
            if s.lance_phase_t >= end_sec:
                s.lance_phase = PHASE_IDLE
                s.lance_spent = True


def weapons_suppressed(s: SimShip) -> bool:
    return s.has_lance and s.lance_phase != PHASE_IDLE
