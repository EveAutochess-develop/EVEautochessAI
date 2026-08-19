"""Combat drone / excavator / FAX / fighter spawn — combat_resolver.gd §14C."""

from __future__ import annotations

import math
from typing import Any

from eveac_ai.content import Content
from eveac_ai.ship import SimShip, spawn_from_content

RACE_DRONE_LIGHT = {
    "amarr": 1001, "caldari": 1002, "gallente": 1003, "minmatar": 1004,
    "blood": 1001, "sansha": 1001, "mordu": 1002, "serpentis": 1003, "soe": 1003, "angel": 1004,
    "guristas": 1502,
}
RACE_DRONE_MEDIUM = {
    "amarr": 1005, "caldari": 1006, "gallente": 1007, "minmatar": 1008,
    "blood": 1005, "sansha": 1005, "mordu": 1006, "serpentis": 1007, "soe": 1007, "angel": 1008,
    "guristas": 1506,
}
RACE_DRONE_HEAVY = {
    "amarr": 1011, "caldari": 1012, "gallente": 1013, "minmatar": 1014,
    "blood": 1011, "sansha": 1011, "mordu": 1012, "serpentis": 1013, "soe": 1013, "angel": 1014,
    "guristas": 1512,
}
DRONE_COUNT_EXCEPTIONS = {42: 5, 44: 4, 55: 4, 56: 5}
DRONE_CAP = 5
DRONE_BW_COST = 5.0
DRONE_REVIVE_DELAY_S = 400.0
TAU = math.tau


def _i(v: Any, d: int = 0) -> int:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return d


def _ids(raw: Any) -> list[int]:
    out: list[int] = []
    if isinstance(raw, list):
        for v in raw:
            n = _i(v, 0)
            if n > 0:
                out.append(n)
    return out


def heavy_repair_drone_unit_ids(data: dict[str, Any]) -> list[int]:
    arr = _ids(data.get("heavy_repair_drone_ids"))
    if arr:
        return arr
    one = _i(data.get("heavy_repair_drone_id"), 0)
    if one <= 0:
        return []
    n = max(_i(data.get("heavy_repair_drone_count"), 4), 0)
    return [one] * n


def carrier_fighter_unit_ids(data: dict[str, Any]) -> list[int]:
    arr = _ids(data.get("fighter_unit_ids"))
    if arr:
        return arr
    one = _i(data.get("fighter_unit_id"), 0)
    return [one] if one > 0 else []


def drone_spawn_policy(hull: dict[str, Any], *, race: str, group: str, ship_id: int, is_logistic: bool) -> dict[str, Any]:
    if heavy_repair_drone_unit_ids(hull):
        return {"count": 0, "drone_id": 0, "drone_ids": []}
    unit_ids = _ids(hull.get("drone_unit_ids"))
    if unit_ids:
        return {"count": len(unit_ids), "drone_ids": unit_ids, "drone_id": unit_ids[0]}
    mining_drone_id = _i(hull.get("mining_drone_id"), 0)
    if mining_drone_id > 0:
        mcount = _i(hull.get("drone_bay_slots"), _i(hull.get("drone_count_cap"), 0))
        if mcount <= 0:
            mcount = _i(hull.get("mining_drone_count"), 4)
        return {"count": mcount, "drone_id": mining_drone_id, "drone_ids": []}
    if is_logistic and group in ("cruiser", "battlecruiser"):
        return {"count": 0, "drone_id": 0, "drone_ids": []}
    if ship_id in DRONE_COUNT_EXCEPTIONS:
        cnt = DRONE_COUNT_EXCEPTIONS[ship_id]
        if group == "battlecruiser":
            return {"count": cnt, "drone_id": int(RACE_DRONE_MEDIUM.get(race, 1005)), "drone_ids": []}
        if group == "battleship":
            return {"count": cnt, "drone_id": int(RACE_DRONE_HEAVY.get(race, 1011)), "drone_ids": []}
    if group == "battlecruiser":
        return {"count": 1, "drone_id": int(RACE_DRONE_MEDIUM.get(race, 1005)), "drone_ids": []}
    if group == "battleship":
        if is_logistic:
            return {"count": 0, "drone_id": 0, "drone_ids": []}
        return {"count": 2, "drone_id": int(RACE_DRONE_HEAVY.get(race, 1011)), "drone_ids": []}
    slots = _i(hull.get("drone_bay_slots"), _i(hull.get("drone_count_cap"), 0))
    if slots <= 0 and "drone_bay_slots" not in hull:
        bw = float(hull.get("drone_bandwidth") or 0)
        if bw > 0:
            slots = int(bw // DRONE_BW_COST)
    if slots <= 0:
        return {"count": 0, "drone_id": 0, "drone_ids": []}
    if group == "mining_barge":
        return {"count": slots, "drone_id": 1007, "drone_ids": []}
    if group == "industrial_command":
        return {"count": slots, "drone_id": 1013, "drone_ids": []}
    if group == "cruiser" and bool(hull.get("faction_ship")):
        return {"count": slots, "drone_id": int(RACE_DRONE_MEDIUM.get(race, 1005)), "drone_ids": []}
    if race == "guristas":
        if group == "cruiser":
            return {"count": slots, "drone_id": 1506, "drone_ids": []}
        if group == "battleship":
            return {"count": slots, "drone_id": 1512, "drone_ids": []}
        return {"count": slots, "drone_id": 1502, "drone_ids": []}
    return {"count": slots, "drone_id": int(RACE_DRONE_LIGHT.get(race, 1001)), "drone_ids": []}


def _place_child(content: Content, mother: SimShip, drone_id: int, uid: int, ang: float, rad: float, y: float, star: int) -> SimShip:
    d = spawn_from_content(
        content,
        str(drone_id),
        mother.team,
        uid,
        star=star,
        x=mother.x + math.cos(ang) * rad,
        z=mother.z + math.sin(ang) * rad,
        y=mother.y + y,
    )
    d.mother_uid = mother.uid
    d.orbit_phase = ang
    return d


def revive_drone(content: Content, mother: SimShip, drone_id: int, uid: int, star: int, have_n: int) -> SimShip:
    ang = float(have_n) * TAU / float(max(have_n + 1, 1))
    d = _place_child(content, mother, drone_id, uid, ang, 1.15, 0.2, star)
    d.orbit_dir = 1.0 if (have_n % 2) == 0 else -1.0
    return d


def spawn_combat_unmanned(content: Content, ships: list[SimShip], next_uid: int) -> int:
    hulls = [s for s in ships if not s.is_unmanned and s.alive()]
    uid = next_uid
    extra: list[SimShip] = []
    for s in hulls:
        hull = content.ships.get(str(s.ship_id)) or {}
        role = str(hull.get("capital_role") or s.capital_role or "").lower()
        if role in ("carrier", "force_auxiliary", "dreadnought"):
            continue
        if heavy_repair_drone_unit_ids(hull):
            continue
        pol = drone_spawn_policy(hull, race=s.race, group=s.ship_group, ship_id=_i(s.ship_id, 0), is_logistic=s.is_logistic)
        n = min(DRONE_CAP, int(pol.get("count") or 0))
        if n <= 0:
            continue
        id_list = list(pol.get("drone_ids") or [])
        drone_id = int(pol.get("drone_id") or 0)
        for i in range(n):
            sid = drone_id
            if id_list:
                sid = int(id_list[i % len(id_list)])
            if sid <= 0:
                continue
            ang = float(i) * TAU / float(max(1, n))
            d = _place_child(content, s, sid, uid, ang, 1.15, 0.2, s.star)
            d.orbit_dir = 1.0 if (i % 2) == 0 else -1.0
            extra.append(d)
            uid += 1
    for s in hulls:
        hull = content.ships.get(str(s.ship_id)) or {}
        fighter_ids = carrier_fighter_unit_ids(hull)
        role = str(hull.get("capital_role") or s.capital_role or "").lower()
        if role == "carrier" or fighter_ids:
            if not fighter_ids:
                continue
            active_max = max(1, _i(hull.get("fighter_squadrons"), 3))
            if len(fighter_ids) > 1:
                active_max = max(active_max, len(fighter_ids))
            tubes = max(1, _i(hull.get("fighter_tubes_per_squadron"), 3))
            for ac in range(active_max):
                fid = int(fighter_ids[ac % len(fighter_ids)])
                for i in range(tubes):
                    ang = float(ac * tubes + i) * TAU / float(active_max * tubes)
                    d = _place_child(content, s, fid, uid, ang, 1.4, 0.25, s.star)
                    d.orbit_dir = 1.0 if (i % 2) == 0 else -1.0
                    extra.append(d)
                    uid += 1
            continue
        repair_ids = heavy_repair_drone_unit_ids(hull)
        if role == "force_auxiliary" or repair_ids:
            n_slots = len(repair_ids)
            for j, did in enumerate(repair_ids):
                if did <= 0:
                    continue
                ang = float(j) * TAU / float(max(1, n_slots))
                d = _place_child(content, s, did, uid, ang, 1.6, 0.25, 1)
                extra.append(d)
                uid += 1
    ships.extend(extra)
    return uid


def orbit_plane_basis(tilt_deg: float, az: float) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    tilt = math.radians(max(0.0, min(89.5, tilt_deg)))
    lean = (math.cos(az), 0.0, math.sin(az))
    nx = 0.0 * math.cos(tilt) + lean[0] * math.sin(tilt)
    ny = 1.0 * math.cos(tilt) + lean[1] * math.sin(tilt)
    nz = 0.0 * math.cos(tilt) + lean[2] * math.sin(tilt)
    nlen = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
    nx, ny, nz = nx / nlen, ny / nlen, nz / nlen
    e1x, e1y, e1z = ny * 0.0 - nz * 1.0, nz * 0.0 - nx * 0.0, nx * 1.0 - ny * 0.0
    e1l = math.sqrt(e1x * e1x + e1y * e1y + e1z * e1z)
    if e1l < 1e-8:
        e1x, e1y, e1z = lean[1] * nz - lean[2] * ny, lean[2] * nx - lean[0] * nz, lean[0] * ny - lean[1] * nx
        e1l = math.sqrt(e1x * e1x + e1y * e1y + e1z * e1z) or 1.0
    e1x, e1y, e1z = e1x / e1l, e1y / e1l, e1z / e1l
    e2x = ny * e1z - nz * e1y
    e2y = nz * e1x - nx * e1z
    e2z = nx * e1y - ny * e1x
    e2l = math.sqrt(e2x * e2x + e2y * e2y + e2z * e2z) or 1.0
    return (e1x, e1y, e1z), (e2x / e2l, e2y / e2l, e2z / e2l)


def step_orbit(d: SimShip, cx: float, cy: float, cz: float, dt: float, radius: float, rng_tilt: float, rng_az: float) -> None:
    if d.orbit_tilt < 0.0:
        d.orbit_tilt = 20.0 + rng_tilt * 69.0
        d.orbit_az = rng_az * TAU
    if abs(d.orbit_dir) < 0.5:
        d.orbit_dir = 1.0
    e1, e2 = orbit_plane_basis(d.orbit_tilt, d.orbit_az)
    omega = max(0.4, min(2.5, d.speed / max(80.0, radius * 400.0)))
    d.orbit_phase += d.orbit_dir * omega * dt
    c, s = math.cos(d.orbit_phase), math.sin(d.orbit_phase)
    d.x = cx + radius * (e1[0] * c + e2[0] * s)
    d.y = cy + radius * (e1[1] * c + e2[1] * s)
    d.z = cz + radius * (e1[2] * c + e2[2] * s)
