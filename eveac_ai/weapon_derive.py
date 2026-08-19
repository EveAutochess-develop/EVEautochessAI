"""Manned hull attack = slots × representative module. SHIP_STATS_V2 §2.2 / ship_weapon_derive.gd."""

from __future__ import annotations

import math
from typing import Any

WEAPON_KIT: dict[str, dict[str, int]] = {
    "laser": {"frigate": 453, "destroyer": 453, "cruiser": 456, "large": 462, "capital": 11002810000},
    "rail": {"frigate": 561, "destroyer": 561, "cruiser": 570, "large": 574, "capital": 11000320000},
    "cannon": {"frigate": 485, "destroyer": 485, "cruiser": 491, "large": 498, "capital": 11004810000},
    "missile": {"frigate": 499, "destroyer": 499, "cruiser": 501, "large": 13320, "capital": 11023000000},
}
REPAIR_KIT: dict[str, list[int]] = {
    "amarr": [11355, 11357, 11359],
    "caldari": [3586, 3596, 3606],
    "gallente": [27932, 27930, 27904],
    "minmatar": [11355, 11357, 11359],
}
DEFAULT_KIT_METERS_PER_CELL = 500.0
ZERO = {"emp": 0.0, "thermal": 0.0, "kinetic": 0.0, "explosive": 0.0}


def _i(v: Any, d: int = 0) -> int:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return d


def _f(v: Any, d: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def uses_baked_star_attack(ship: dict[str, Any]) -> bool:
    return bool(ship.get("is_unmanned"))


def guns_muted(ship: dict[str, Any]) -> bool:
    fx = str(ship.get("weapon_fx") or "")
    role = str(ship.get("capital_role") or "")
    return role == "carrier" or fx == "mining" or bool(ship.get("is_mining_ship"))


def get_module(content: Any, mod_id: int) -> dict[str, Any]:
    if mod_id <= 0:
        return {}
    mods = getattr(content, "modules", None) or {}
    return mods.get(str(mod_id)) or mods.get(mod_id) or {}


def should_derive(ship: dict[str, Any], content: Any) -> bool:
    if uses_baked_star_attack(ship):
        return False
    if guns_muted(ship):
        return True
    fx = str(ship.get("weapon_fx") or "")
    logistic = bool(ship.get("is_logistic")) or fx == "heal"
    if logistic:
        rid = resolve_repair_module_id(ship)
        if rid > 0 and get_module(content, rid):
            return True
    explicit = _i(ship.get("source_module_type_id"), 0)
    if explicit > 0:
        return bool(get_module(content, explicit))
    mid = resolve_module_id(ship)
    if mid <= 0:
        return True
    return bool(get_module(content, mid))


def attack_slot_count(ship: dict[str, Any]) -> int:
    n = _i(ship.get("attack_weapon_slots"), 0)
    if n <= 0:
        n = _i(ship.get("hi_slots"), 0)
    return max(n, 0)


def kit_meters_per_cell(content: Any) -> float:
    combat = getattr(content, "combat", None) or {}
    v = _f(combat.get("weapon_kit_meters_per_cell"), 0.0)
    if v > 0.0:
        return v
    v = _f(combat.get("speed_meters_per_cell"), 0.0)
    return v if v > 0.0 else DEFAULT_KIT_METERS_PER_CELL


def meters_to_cells(meters: float, content: Any) -> float:
    return round(float(meters) / kit_meters_per_cell(content), 3)


def size_key(ship: dict[str, Any]) -> str:
    tier = str(ship.get("weapon_tier") or "")
    if tier == "small":
        return "frigate"
    if tier == "large":
        return "large"
    if tier == "medium":
        return "cruiser"
    if tier == "capital":
        return "capital"
    group = str(ship.get("ship_group") or "")
    if group in ("frigate", "destroyer"):
        return "frigate"
    if group in ("dreadnought", "carrier", "force_auxiliary", "titan"):
        return "capital"
    if group == "battleship":
        return "large"
    if group in ("cruiser", "battlecruiser"):
        return "cruiser"
    return "frigate"


def resolve_module_id(ship: dict[str, Any]) -> int:
    mod_id = _i(ship.get("source_module_type_id"), 0)
    if mod_id > 0:
        return mod_id
    fx = str(ship.get("weapon_fx") or "")
    if fx == "heal":
        race = str(ship.get("race") or "amarr").lower()
        fx = {"amarr": "laser", "caldari": "rail", "minmatar": "cannon", "gallente": "rail"}.get(race, "laser")
    kit = WEAPON_KIT.get(fx) or {}
    return int(kit.get(size_key(ship), 0) or 0)


def resolve_repair_module_id(ship: dict[str, Any]) -> int:
    rid = _i(ship.get("source_repair_module_type_id"), 0)
    if rid > 0:
        return rid
    race = str(ship.get("race") or "amarr").lower()
    arr = REPAIR_KIT.get(race) or REPAIR_KIT["amarr"]
    if len(arr) < 3:
        return 0
    key = size_key(ship)
    if key in ("large", "capital"):
        return arr[2]
    if key == "cruiser":
        return arr[1]
    return arr[0]


def _scale_damage(d: dict[str, float], mul: float) -> dict[str, float]:
    return {k: round(_f(d.get(k)) * mul, 2) for k in ZERO}


def _per_slot_weapon(content: Any, mod_id: int) -> dict[str, Any]:
    out: dict[str, Any] = {
        "damage": dict(ZERO),
        "tracking": 0.0,
        "optimal": 0.0,
        "falloff": 0.0,
        "optimal_sig_radius": 40.0,
        "rate_of_fire_s": 1.0,
        "explosion_radius": 0.0,
        "explosion_velocity": 0.0,
        "drf": 0.0,
    }
    mod = get_module(content, mod_id)
    if not mod:
        return out
    out["damage"] = {
        "emp": _f(mod.get("emDamage")),
        "thermal": _f(mod.get("thermalDamage")),
        "kinetic": _f(mod.get("kineticDamage")),
        "explosive": _f(mod.get("explosiveDamage")),
    }
    out["tracking"] = _f(mod.get("trackingSpeed"))
    out["optimal"] = meters_to_cells(_f(mod.get("maxRange")), content)
    out["falloff"] = meters_to_cells(_f(mod.get("falloff")), content)
    out["optimal_sig_radius"] = _f(mod.get("signatureResolution"), 40.0)
    out["rate_of_fire_s"] = round(_f(mod.get("rateOfFire"), 1000.0) / 1000.0, 3)
    out["explosion_radius"] = _f(mod.get("explosionRadius"))
    out["explosion_velocity"] = _f(mod.get("explosionVelocity"))
    out["drf"] = _f(mod.get("aoeDamageReductionFactor"))
    return out


def _racial_repair(amount: float, race: str) -> dict[str, float]:
    if amount <= 0.0:
        return {"shield": 0.0, "armor": 0.0, "structure": 0.0}
    r = race.lower()
    if r == "amarr":
        return {"shield": 0.0, "armor": amount, "structure": 0.0}
    if r == "caldari":
        return {"shield": amount, "armor": 0.0, "structure": 0.0}
    if r == "gallente":
        return {"shield": 0.0, "armor": 0.0, "structure": amount}
    half = amount / 2.0
    return {"shield": float(math.ceil(half)), "armor": math.floor(half), "structure": 0.0}


def derive_attack(ship: dict[str, Any], content: Any) -> dict[str, Any]:
    if not should_derive(ship, content):
        return {}
    fx = str(ship.get("weapon_fx") or "")
    slots = attack_slot_count(ship)
    mute = guns_muted(ship)
    mid = resolve_module_id(ship)
    if mute or mid <= 0:
        wpn = {
            "damage": dict(ZERO),
            "tracking": 0.0,
            "optimal": 0.0,
            "falloff": 0.0,
            "optimal_sig_radius": 40.0,
            "rate_of_fire_s": _f(ship.get("attack_cycle_s"), 1.0),
            "explosion_radius": 0.0,
            "explosion_velocity": 0.0,
            "drf": 0.0,
        }
    else:
        wpn = _per_slot_weapon(content, mid)
    slot_dmg = wpn.get("damage") or ZERO
    total_1 = {k: _f(slot_dmg.get(k)) * float(slots) for k in ZERO}
    out: dict[str, Any] = {
        "damage": _scale_damage(total_1, 1.0),
        "tracking": _f(wpn.get("tracking")),
        "optimal": _f(wpn.get("optimal")),
        "falloff": _f(wpn.get("falloff")),
        "optimal_sig_radius": _f(wpn.get("optimal_sig_radius"), 40.0),
        "explosion_radius": _f(wpn.get("explosion_radius")),
        "explosion_velocity": _f(wpn.get("explosion_velocity")),
        "drf": _f(wpn.get("drf")),
        "repair": {"shield": 0.0, "armor": 0.0, "structure": 0.0},
        "attack_cycle_s": _f(wpn.get("rate_of_fire_s"), -1.0),
    }
    logistic = bool(ship.get("is_logistic")) or fx == "heal"
    if logistic:
        rid = resolve_repair_module_id(ship)
        amount = 0.0
        cycle_s = _f(wpn.get("rate_of_fire_s"))
        opt = _f(wpn.get("optimal"))
        rmod = get_module(content, rid) if rid > 0 else {}
        if rmod:
            amount = _f(rmod.get("structureDamageAmount"), _f(rmod.get("armorDamageAmount"), _f(rmod.get("shieldBonus"))))
            dur_ms = _f(rmod.get("duration"), _f(rmod.get("rateOfFire"), 3000.0))
            cycle_s = round(dur_ms / 1000.0, 3)
            opt = meters_to_cells(_f(rmod.get("maxRange")), content)
        hi = _i(ship.get("hi_slots"), slots)
        out["repair"] = _racial_repair(amount * float(max(hi, 0)), str(ship.get("race") or "amarr"))
        if opt > 0.0:
            out["optimal"] = opt
        if cycle_s > 0.0:
            out["attack_cycle_s"] = cycle_s
    return out


def merge_into_star(ship: dict[str, Any], star_row: dict[str, Any], content: Any) -> dict[str, Any]:
    out = dict(star_row)
    if not star_row or not should_derive(ship, content):
        return out
    derived = derive_attack(ship, content)
    if not derived:
        return out
    out["damage"] = derived["damage"]
    out["tracking"] = derived["tracking"]
    out["optimal"] = derived["optimal"]
    out["falloff"] = derived["falloff"]
    out["optimal_sig_radius"] = derived["optimal_sig_radius"]
    out["repair"] = derived["repair"]
    if _f(derived.get("explosion_radius")) > 0.0 or str(ship.get("weapon_fx") or "") == "missile":
        out["explosion_radius"] = derived["explosion_radius"]
        out["explosion_velocity"] = derived["explosion_velocity"]
        out["drf"] = derived["drf"]
    out["_attack_cycle_s"] = derived["attack_cycle_s"]
    reach = _f(derived.get("optimal")) + _f(derived.get("falloff"))
    baked = _f(out.get("attack_range"))
    if reach > baked:
        out["attack_range"] = reach
    return out


def synthesize_unmanned_star(star1: dict[str, Any], star: int) -> dict[str, Any]:
    row = dict(star1)
    mul = float(max(star, 1))
    if mul <= 1.001:
        return row
    for k in ("shield_hp", "armor_hp", "structure_hp"):
        if k in row:
            row[k] = _f(row[k]) * mul
    dmg = row.get("damage")
    if isinstance(dmg, dict):
        row["damage"] = {dk: _f(dv) * mul for dk, dv in dmg.items()}
    return row
