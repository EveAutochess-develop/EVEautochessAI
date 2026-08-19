from __future__ import annotations

import math
from typing import Any


def distance_meters(cells: float, meters_per_cell: float = 2000.0) -> float:
    return cells * meters_per_cell


def tracking_distance_meters(cells: float, tracking_meters_per_cell: float = 2000.0) -> float:
    return cells * tracking_meters_per_cell


def turret_hit_chance(
    attacker_tracking: float,
    attacker_optimal_cells: float,
    attacker_falloff_cells: float,
    attacker_optimal_sig: float,
    target_speed: float,
    target_signature: float,
    distance_cells: float,
    *,
    meters_per_cell: float = 2000.0,
    tracking_meters_per_cell: float = 2000.0,
    hit_chance_min: float = 0.01,
    hit_chance_max: float = 0.99,
) -> float:
    if attacker_tracking <= 0.0:
        return 0.0
    d_track = tracking_distance_meters(distance_cells, tracking_meters_per_cell)
    d_m = distance_meters(distance_cells, meters_per_cell)
    v = max(target_speed, 0.0)
    omega = v / max(d_track, 1.0)
    sig_res = max(attacker_optimal_sig, 1.0)
    sig_tgt = max(target_signature, 1.0)
    tracking_term = (omega / attacker_tracking) * (sig_res / sig_tgt)
    r_opt = distance_meters(attacker_optimal_cells, meters_per_cell)
    r_fo = max(distance_meters(attacker_falloff_cells, meters_per_cell), 1.0)
    range_term = max(0.0, d_m - r_opt) / r_fo
    exponent = tracking_term * tracking_term + range_term * range_term
    p = 0.5 ** exponent
    return max(hit_chance_min, min(hit_chance_max, p))


def turret_hit_quality(x: float, p_hit: float) -> float:
    if x > p_hit:
        return 0.0
    if x < 0.01:
        return 3.0
    return x + 0.5


def missile_damage_factor(
    target_signature: float,
    target_speed: float,
    explosion_radius: float,
    explosion_velocity: float,
    drf: float,
    drs: float,
    *,
    missile_drs_default: float = 5.5,
) -> float:
    er = max(explosion_radius, 1.0)
    ev = max(explosion_velocity, 1.0)
    sig = max(target_signature, 1.0)
    vt = max(target_speed, 0.0)
    sig_ratio = sig / er
    if drf <= 0.0:
        return min(1.0, sig_ratio)
    drf_exp = drf
    if drf > 1.0:
        drs_eff = drs
        if drs_eff <= 1.0:
            drs_eff = missile_drs_default if missile_drs_default > 1.0 else 5.5
        drf_exp = math.log(drf) / math.log(drs_eff)
    vt_eff = max(vt, 0.0001)
    speed_term = (sig_ratio * (ev / vt_eff)) ** drf_exp
    return min(1.0, min(sig_ratio, speed_term))


def lock_time_s(scan_resolution: float, target_signature: float, lock_time_constant: float = 40000.0) -> float:
    scan = max(scan_resolution, 1.0)
    sig = max(target_signature, 1.0)
    return lock_time_constant / (scan * sig)


def combat_params(combat: dict[str, Any]) -> dict[str, float]:
    def f(k: str, d: float) -> float:
        try:
            return float(combat.get(k, d))
        except (TypeError, ValueError):
            return d

    return {
        "meters_per_cell": f("meters_per_cell", 2000.0),
        "tracking_meters_per_cell": f("tracking_meters_per_cell", 2000.0),
        "hit_chance_min": f("hit_chance_min", 0.01),
        "hit_chance_max": f("hit_chance_max", 0.99),
        "lock_time_constant": f("lock_time_constant", 40000.0),
        "missile_speed_cells_per_s": f("missile_speed_cells_per_s", 1.5),
        "missile_hit_radius_wu": f("missile_hit_radius_wu", 0.45),
        "world_units_per_cell": f("world_units_per_cell", 3.0),
        "missile_drs_default": f("missile_drs_default", 5.5),
        "move_speed": f("move_speed", 3.5),
        "move_speed_scale": f("move_speed_scale", 1.65),
        "cap_disable_attack_function_pct": f("cap_disable_attack_function_pct", 0.10),
        "cap_drain_fraction_per_cycle": f("cap_drain_fraction_per_cycle", 0.02),
        "attack_duration_s": f("attack_duration_s", 1.0),
        "logistic_attack_duration_s": f("logistic_attack_duration_s", 2.0),
        "attack_cycle_cap_s": f("attack_cycle_cap_s", 6.0),
        "min_engagement_cells": f("min_engagement_cells", 1.0),
        "approach_factor": f("approach_factor", 0.9),
    }
