from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from eveac_ai.content import Content, star_at
from eveac_ai.formulas import combat_params
from eveac_ai.weapon_derive import merge_into_star, should_derive


DTYPES = ("emp", "thermal", "kinetic", "explosive")


def _f(v: Any, d: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def _resist(src: dict[str, Any] | None) -> dict[str, float]:
    src = src or {}
    return {k: max(0.0, min(0.95, _f(src.get(k, 0.0)))) for k in DTYPES}


def damage_sum(dmg: dict[str, float]) -> float:
    return sum(max(0.0, dmg.get(k, 0.0)) for k in DTYPES)


@dataclass
class SimShip:
    uid: int
    ship_id: str
    team: int
    star: int = 1
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    race: str = "amarr"
    is_logistic: bool = False
    is_unmanned: bool = False
    is_mining_ship: bool = False
    is_protect_target: bool = False
    unmanned_kind: str = ""
    weapon_tier: str = ""
    ship_group: str = ""
    capital_role: str = ""
    mother_uid: int = 0
    skip_unmanned: bool = False
    chase_y: bool = False
    orbit_phase: float = 0.0
    orbit_dir: float = 1.0
    orbit_tilt: float = -1.0
    orbit_az: float = 0.0
    no_target_acc: float = 0.0
    stall_until: float = 0.0
    stall_uid: int = 0
    stall_sh: float = 0.0
    stall_ar: float = 0.0
    stall_st: float = 0.0
    weapon_fx: str = "laser"
    cost: float = 5.0
    shield: float = 1.0
    armor: float = 1.0
    structure: float = 1.0
    max_shield: float = 1.0
    max_armor: float = 1.0
    max_structure: float = 1.0
    shield_resist: dict[str, float] = field(default_factory=dict)
    armor_resist: dict[str, float] = field(default_factory=dict)
    structure_resist: dict[str, float] = field(default_factory=dict)
    damage: dict[str, float] = field(default_factory=dict)
    repair: dict[str, float] = field(default_factory=dict)
    tracking: float = 0.08
    optimal_cells: float = 8.0
    falloff_cells: float = 4.0
    optimal_sig: float = 40.0
    explosion_radius: float = 0.0
    explosion_velocity: float = 0.0
    missile_drf: float = 0.0
    missile_drs: float = 5.5
    signature: float = 40.0
    scan: float = 400.0
    speed: float = 300.0
    attack_range: float = 6.0
    attack_duration: float = 1.0
    cap_capacity: float = 0.0
    cap_recharge_s: float = 1.0
    cap_current: float = 0.0
    cap_cost: float = -1.0
    last_attack: float = -999.0
    lock_uid: int = 0
    lock_timer: float = 0.0
    lock_need: float = 0.0
    destroyed: bool = False
    dmg_out: float = 0.0
    dmg_in: float = 0.0
    repaired: float = 0.0
    cap_used: float = 0.0
    lock_s: float = 0.0
    hold_until: float = 0.0
    speed_base: float = 300.0
    heal_recv_mul: float = 1.0
    heal_debuff_until: float = 0.0
    speed_debuff_until: float = 0.0
    has_lance: bool = False
    lance_spent: bool = False
    lance_phase: int = 0
    lance_phase_t: float = 0.0
    lance_tick_acc: float = 0.0
    lance_ox: float = 0.0
    lance_oy: float = 0.0
    lance_oz: float = 0.0
    lance_dx: float = 0.0
    lance_dy: float = 0.0
    lance_dz: float = 1.0
    lance_beam_h: float = 20.0
    last_repair: float = -999.0

    def alive(self) -> bool:
        return not self.destroyed and self.structure > 0.0

    def has_offense(self) -> bool:
        return (not self.is_logistic) and damage_sum(self.damage) > 0.001

    def is_missile(self) -> bool:
        return self.weapon_fx == "missile"

    def dist_to(self, o: "SimShip") -> float:
        dx = self.x - o.x
        dy = self.y - o.y
        dz = self.z - o.z
        return (dx * dx + dy * dy + dz * dz) ** 0.5

    def cap_frac(self) -> float:
        if self.cap_capacity <= 0.0:
            return 1.0
        return self.cap_current / self.cap_capacity


def is_protect_hull(hull: dict[str, Any] | None) -> bool:
    hull = hull or {}
    if str(hull.get("ship_group") or "").lower() == "freighter":
        return True
    tags = [str(t).lower() for t in (hull.get("tags") or [])]
    return "pve_salvage" in tags


def spawn_from_content(
    content: Content, ship_id: str, team: int, uid: int, star: int = 1, x: float = 0.0, z: float = 0.0, y: float = 0.0
) -> SimShip:
    hull = content.ships.get(str(ship_id)) or {}
    st = merge_into_star(hull, star_at(hull, star), content)
    cp = combat_params(content.combat)
    dmg = st.get("damage") if isinstance(st.get("damage"), dict) else {}
    emp = _f(dmg.get("emp"), 0.0)
    th = _f(dmg.get("thermal"), 0.0)
    kn = _f(dmg.get("kinetic"), 0.0)
    ex = _f(dmg.get("explosive"), 0.0)
    cost = _f(hull.get("cost"), 5.0)
    logistic = bool(hull.get("is_logistic") or st.get("is_logistic"))
    derived = should_derive(hull, content)
    star_dph = float(max(star, 1)) if derived else 1.0
    emp *= star_dph
    th *= star_dph
    kn *= star_dph
    ex *= star_dph
    derived_cycle = _f(st.get("_attack_cycle_s"), -1.0)
    cycle = derived_cycle if (derived and derived_cycle > 0.0) else _f(hull.get("attack_cycle_s"), 0.0)
    if cycle <= 0.0:
        cycle = cp["logistic_attack_duration_s"] if logistic else cp["attack_duration_s"]
    role = str(hull.get("capital_role") or "")
    if not role and not hull.get("requires_cyno_entry"):
        cycle = min(cycle, cp["attack_cycle_cap_s"]) if cycle > 0 else cp["attack_duration_s"]
    unmanned = bool(hull.get("is_unmanned", False))
    sh = _f(st.get("shield_hp"), 200.0)
    ar = _f(st.get("armor_hp"), 200.0)
    su = _f(st.get("structure_hp"), max(50.0, ar * 0.5))
    if unmanned:
        sh *= 0.5
        ar *= 0.5
        su *= 0.5
    cap = _f(hull.get("capacitor_capacity"), 0.0)
    s = SimShip(
        uid=uid,
        ship_id=str(ship_id),
        team=team,
        star=star,
        x=x,
        y=y,
        z=z,
        race=str(hull.get("race", "amarr")).lower(),
        is_logistic=logistic,
        is_unmanned=unmanned,
        is_mining_ship=bool(hull.get("is_mining_ship")),
        is_protect_target=is_protect_hull(hull),
        unmanned_kind=str(hull.get("unmanned_kind") or ""),
        weapon_tier=str(hull.get("weapon_tier") or "").lower(),
        ship_group=str(hull.get("ship_group") or "").lower(),
        capital_role=str(hull.get("capital_role") or "").lower(),
        weapon_fx=str(hull.get("weapon_fx", "laser")),
        cost=cost,
        shield=sh,
        armor=ar,
        structure=su,
        max_shield=sh,
        max_armor=ar,
        max_structure=su,
        shield_resist=_resist(st.get("shield_resist") if isinstance(st.get("shield_resist"), dict) else {}),
        armor_resist=_resist(st.get("armor_resist") if isinstance(st.get("armor_resist"), dict) else {}),
        structure_resist=_resist(st.get("structure_resist") if isinstance(st.get("structure_resist"), dict) else st.get("armor_resist")),
        damage={"emp": emp, "thermal": th, "kinetic": kn, "explosive": ex},
        repair={
            "shield": _f((st.get("repair") or {}).get("shield"), 0.0) if isinstance(st.get("repair"), dict) else 0.0,
            "armor": _f((st.get("repair") or {}).get("armor"), 0.0) if isinstance(st.get("repair"), dict) else 0.0,
            "structure": _f((st.get("repair") or {}).get("structure"), 0.0) if isinstance(st.get("repair"), dict) else 0.0,
        },
        tracking=max(_f(st.get("tracking"), 0.0), 0.0),
        optimal_cells=_f(st.get("optimal"), 0.0),
        falloff_cells=_f(st.get("falloff"), 0.0),
        optimal_sig=_f(st.get("optimal_sig_radius"), 40.0),
        explosion_radius=_f(st.get("explosion_radius"), 0.0),
        explosion_velocity=_f(st.get("explosion_velocity"), 0.0),
        missile_drf=_f(st.get("drf"), 0.0),
        missile_drs=_f(st.get("drs"), cp["missile_drs_default"]),
        signature=max(_f(hull.get("signature_radius"), 40.0), 1.0),
        scan=max(_f(hull.get("scan_resolution"), 400.0), 1.0),
        speed=_f(hull.get("speed"), 300.0),
        attack_range=_f(st.get("attack_range"), 6.0),
        attack_duration=cycle,
        cap_capacity=cap,
        cap_recharge_s=max(_f(hull.get("capacitor_recharge_s"), 1.0), 0.001),
        cap_current=cap,
        cap_cost=_f(st.get("cap_cost"), _f(hull.get("cap_cost"), -1.0)),
        speed_base=_f(hull.get("speed"), 300.0),
    )
    fx = s.weapon_fx.lower()
    s.skip_unmanned = s.weapon_tier in ("large", "capital") and fx in ("laser", "rail", "cannon", "missile")
    s.chase_y = s.ship_group in ("frigate", "destroyer")
    if s.optimal_cells > 500:
        s.optimal_cells = s.attack_range
    if s.falloff_cells > 500:
        s.falloff_cells = max(2.0, s.attack_range * 0.5)
    return s


def apply_hit(ship: SimShip, dmg: dict[str, float], pierce: bool = True) -> dict[str, Any]:
    if ship.destroyed:
        return {"destroyed": True, "dealt": 0.0}
    remaining = {k: max(0.0, float(dmg.get(k, 0.0))) for k in DTYPES}
    if damage_sum(remaining) <= 0.0:
        return {"destroyed": False, "dealt": 0.0}
    applied = 0.0
    layers = (
        ("shield", "shield", ship.shield_resist),
        ("armor", "armor", ship.armor_resist),
        ("structure", "structure", ship.structure_resist),
    )
    for lname, attr, resist_map in layers:
        if damage_sum(remaining) <= 0.0:
            break
        layer_hp = getattr(ship, attr)
        if layer_hp <= 0.0:
            continue
        dealt = 0.0
        for k in DTYPES:
            raw_i = remaining[k]
            if raw_i <= 0.0:
                continue
            resist = resist_map.get(k, 0.0)
            dealt += raw_i * (1.0 - resist)
        if dealt <= 0.0:
            break
        if dealt <= layer_hp:
            setattr(ship, attr, layer_hp - dealt)
            applied += dealt
            remaining = {k: 0.0 for k in DTYPES}
            break
        frac_absorbed = layer_hp / dealt
        if lname == "structure":
            ship.structure -= dealt
        else:
            setattr(ship, attr, 0.0)
        applied += layer_hp
        if lname == "structure" or not pierce:
            remaining = {k: 0.0 for k in DTYPES}
            break
        keep = 1.0 - frac_absorbed
        remaining = {k: remaining[k] * keep for k in DTYPES}
    ship.dmg_in += applied
    if ship.shield <= 0.0 and ship.armor <= 0.0 and ship.structure <= 0.0:
        ship.destroyed = True
        ship.structure = 0.0
        return {"destroyed": True, "dealt": applied}
    return {"destroyed": False, "dealt": applied}


def apply_heal(ship: SimShip, shield: float, armor: float, structure: float) -> float:
    if ship.destroyed:
        return 0.0
    mul = max(0.0, float(ship.heal_recv_mul))
    shield *= mul
    armor *= mul
    structure *= mul
    got = 0.0
    if shield > 0:
        n = min(ship.max_shield - ship.shield, shield)
        ship.shield += n
        got += n
    if armor > 0:
        n = min(ship.max_armor - ship.armor, armor)
        ship.armor += n
        got += n
    if structure > 0:
        n = min(ship.max_structure - ship.structure, structure)
        ship.structure += n
        got += n
    ship.repaired += got
    return got
