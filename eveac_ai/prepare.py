"""Prepare: deploy and fit. Preferences only from genome/priors; rules only for legality."""

from __future__ import annotations

import random
from typing import Any

from eveac_ai.board_view import cols_for_row, field_h, hangar_n
from eveac_ai.content import star_at
from eveac_ai.ranking import is_titan_hull
from eveac_ai.ship import damage_sum

MAX_FIT = 3


def hull_size(hull: dict[str, Any]) -> str:
    g = str(hull.get("ship_group") or "").lower()
    if g in ("carrier", "dreadnought", "force_auxiliary", "capital_industrial", "supercarrier", "titan"):
        return "XL"
    if g in ("battleship",):
        return "L"
    if g in ("cruiser", "battlecruiser", "industrial_command"):
        return "M"
    if g:
        return "S"
    cost = float(hull.get("cost") or 0.0)
    if cost >= 15:
        return "XL"
    if cost >= 10:
        return "L"
    if cost >= 5:
        return "M"
    return "S"


def is_covert_cyno(hull: dict[str, Any] | None) -> bool:
    return str((hull or {}).get("capital_role", "")).lower() == "covert_cyno"


def is_mining_hull(hull: dict[str, Any] | None) -> bool:
    h = hull or {}
    if h.get("is_mining_ship"):
        return True
    return str(h.get("ship_group") or "").lower() in ("mining_barge", "industrial_command", "capital_industrial")


def is_cyno_flagship(hull: dict[str, Any] | None) -> bool:
    return bool((hull or {}).get("requires_cyno_entry"))


def is_wreck_hull(hull: dict[str, Any] | None) -> bool:
    hull = hull or {}
    tags = [str(t).lower() for t in (hull.get("tags") or [])]
    if "wreck" in tags:
        return True
    key = str(hull.get("model_key") or "").lower()
    return "wreck" in key


def is_shop_combat_hull(hull: dict[str, Any] | None) -> bool:
    """Same gate as Godot shop: buyable manned combat hulls only (no titans/wrecks/freighters)."""
    if not hull or not (hull.get("id") is not None or hull.get("name") or hull.get("ship_group")):
        return False
    if is_titan_hull(hull) or is_wreck_hull(hull):
        return False
    if hull.get("shop_eligible") is False:
        return False
    if str(hull.get("ship_group", "")).lower() in ("freighter", "titan"):
        return False
    if hull.get("is_unmanned"):
        return False
    tags = [str(t).lower() for t in (hull.get("tags") or [])]
    if "shop_ineligible" in tags or "pve_salvage" in tags or "wreck" in tags:
        return False
    role = str(hull.get("capital_role", "")).lower()
    if role == "titan":
        return False
    if role == "capital_industrial" and not hull.get("is_mining_ship"):
        return False
    if float(hull.get("cost") or 0.0) <= 0.0:
        return False
    return True


def legal_field_cells(board: dict[str, Any]) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for z in range(field_h(board)):
        for x in range(cols_for_row(board, z)):
            cells.append((x, z))
    return cells


def _prepare_knobs(genome: dict[str, Any]) -> dict[str, Any]:
    p = genome.get("prepare") or {}
    return {
        "buy_n": int(p.get("buy_n", 4)),
        "fit_slots": max(1, min(MAX_FIT, int(p.get("fit_slots", 2)))),
        "hangar_keep_if_economy_ge": float(p.get("hangar_keep_if_economy_ge", 0.22)),
    }


def _ship_w(genome: dict[str, Any], titan: str, sid: str) -> float:
    sl = (genome.get("titan_slices") or {}).get(titan) or {}
    return float((sl.get("ship") or {}).get(str(sid), 0.4))


def _equip_w(genome: dict[str, Any], titan: str, eid: str) -> float:
    sl = (genome.get("titan_slices") or {}).get(titan) or {}
    return float((sl.get("equip") or {}).get(str(eid), 0.45))


def _role_frontness(hull: dict[str, Any] | None) -> float:
    """Combat function vs teammates, not a ship-id lock."""
    hull = hull or {}
    if is_covert_cyno(hull):
        return 0.45
    st = star_at(hull, 1)
    if hull.get("is_logistic") or st.get("is_logistic"):
        return 0.18
    dmg = st.get("damage") if isinstance(st.get("damage"), dict) else {}
    d = damage_sum({k: float(dmg.get(k) or 0) for k in ("emp", "thermal", "kinetic", "explosive")})
    return 0.52 + 0.25 * (d / (d + 80.0))


def _role_speed(hull: dict[str, Any] | None) -> float:
    return float((hull or {}).get("speed") or 0.0)


def _relation_score(
    hull: dict[str, Any],
    x: int,
    z: int,
    placed: list[tuple[dict[str, Any], int, int]],
    stance: dict[str, Any],
    board: dict[str, Any],
) -> float:
    off = float(stance.get("offense") or 0.0)
    form = float(stance.get("formation") or 0.0)
    spd = float(stance.get("speed_control") or 0.0)
    logi = float(stance.get("logistics") or 0.0)
    self_f = _role_frontness(hull)
    self_spd = _role_speed(hull)
    if not placed:
        h = max(1, field_h(board) - 1)
        cols = max(1, cols_for_row(board, z))
        mid = (cols - 1) / 2.0
        z_n = z / h
        return off * z_n + spd * (1.0 - z_n) + form * (1.0 - abs(x - mid) / max(mid, 1.0))
    s = 0.0
    for oh, ox, oz in placed:
        dz = z - oz
        man = abs(x - ox) + abs(z - oz)
        df = self_f - _role_frontness(oh)
        s += off * df * dz
        s += logi * (-df) * dz
        close = 2.0 - min(man, 4)
        s += form * close
        s += spd * (self_spd - _role_speed(oh)) * (abs(x - ox) - abs(z - oz)) * 0.001
    return s


def _place_relative(field_ids: list[str], content: Any, board: dict[str, Any], stance: dict[str, Any]) -> list[tuple[str, int, int]]:
    empty = set(legal_field_cells(board))
    placed_hulls: list[tuple[dict[str, Any], int, int]] = []
    out: list[tuple[str, int, int]] = []
    remain = list(field_ids)
    while remain and empty:
        best: tuple[float, str, int, int] | None = None
        for sid in remain:
            hull = content.ships.get(str(sid)) or {}
            for x, z in empty:
                sc = _relation_score(hull, x, z, placed_hulls, stance, board)
                key = (sc, sid, x, z)
                if best is None or key > (best[0], best[1], -best[2], -best[3]):
                    if best is None or sc > best[0] + 1e-12 or (abs(sc - best[0]) < 1e-12 and (sid, x, z) < (best[1], best[2], best[3])):
                        best = (sc, sid, x, z)
        if best is None:
            break
        _, sid, x, z = best
        remain.remove(sid)
        empty.discard((x, z))
        hull = content.ships.get(str(sid)) or {}
        placed_hulls.append((hull, x, z))
        out.append((sid, x, z))
    return out


def _legal_equip(meta: dict[str, Any], size: str) -> bool:
    if meta.get("shop_pool") is False:
        return False
    if meta.get("implant"):
        return False
    if float(meta.get("cost") or 0.0) <= 0:
        return False
    allowed = [str(x).upper() for x in (meta.get("allowed_on") or [])]
    if allowed and size not in allowed:
        return False
    return True


def assign_equips(content: Any, genome: dict[str, Any], titan: str, pieces: list[dict[str, Any]]) -> None:
    """Greedy: highest equip weight onto highest legal ship weight. Count from prepare.fit_slots."""
    knobs = _prepare_knobs(genome)
    cap = knobs["fit_slots"]
    carriers = [p for p in pieces if not is_covert_cyno(content.ships.get(str(p["ship_id"])))]
    ranked_eq: list[tuple[float, str, str]] = []
    for eid in content.equip_ids:
        meta = content.equip_meta.get(eid) or {}
        ranked_eq.append((_equip_w(genome, titan, eid), eid, str(meta.get("name") or eid)))
    ranked_eq.sort(key=lambda t: (-t[0], t[1]))
    used_on: dict[int, set[str]] = {id(p): set() for p in carriers}
    for p in pieces:
        p["equips"] = []
    for _, eid, name in ranked_eq:
        meta = content.equip_meta.get(eid) or {}
        line = str(meta.get("line") or eid)
        best = None
        best_w = -1.0
        for p in carriers:
            if len(p["equips"]) >= cap:
                continue
            if line in used_on[id(p)]:
                continue
            hull = content.ships.get(str(p["ship_id"])) or {}
            if not _legal_equip(meta, hull_size(hull)):
                continue
            w = _ship_w(genome, titan, p["ship_id"])
            if w > best_w:
                best_w = w
                best = p
        if best is None:
            continue
        best["equips"].append(f"{eid}:{name}")
        used_on[id(best)].add(line)


def pick_equips(content: Any, genome: dict[str, Any], titan: str, ship_id: str, rng: random.Random) -> list[str]:
    dummy = [{"ship_id": ship_id, "equips": []}]
    assign_equips(content, genome, titan, dummy)
    _ = rng
    return dummy[0]["equips"]


def prepare_pieces(
    content: Any,
    genome: dict[str, Any],
    titan: str,
    fleet: list[str],
    board: dict[str, Any],
    rng: random.Random,
    place_net: Any | None = None,
) -> list[dict[str, Any]]:
    _ = rng
    knobs = _prepare_knobs(genome)
    stance = genome.get("stance") or {}
    fleet = [str(s) for s in fleet if is_shop_combat_hull(content.ships.get(str(s)))]
    if not fleet:
        fleet = ["10"]
    flag_ids = [s for s in fleet if is_cyno_flagship(content.ships.get(s))]
    cyno_ids = [s for s in fleet if is_covert_cyno(content.ships.get(s))]
    rest = [s for s in fleet if s not in flag_ids and s not in cyno_ids]
    field_ids = cyno_ids + rest
    hangar_ids = list(flag_ids)
    eco = float(stance.get("economy") or 0.0)
    if eco >= knobs["hangar_keep_if_economy_ge"] and rest:
        park = rest[-1]
        if park in field_ids:
            field_ids = [s for s in field_ids if s != park]
            hangar_ids.append(park)
    hn = hangar_n(board)
    hangar_ids = hangar_ids[:hn]
    pieces: list[dict[str, Any]] = []
    tok = 1
    field_pieces: list[dict[str, Any]] = []
    for sid in field_ids:
        row = {"token": tok, "ship_id": sid, "star": 1, "equips": [], "slot": "field", "x": 0, "z": 0}
        pieces.append(row)
        field_pieces.append(row)
        tok += 1
    for i, sid in enumerate(hangar_ids[:hn]):
        pieces.append({"token": tok, "ship_id": sid, "star": 1, "equips": [], "slot": "hangar", "x": i, "z": 0})
        tok += 1
    assign_equips(content, genome, titan, pieces)
    if place_net is not None and field_pieces:
        place_net.place_field(content, board, stance, field_pieces)
    else:
        laid = _place_relative([p["ship_id"] for p in field_pieces], content, board, stance)
        by = {sid: (x, z) for sid, x, z in laid}
        used: set[tuple[int, int]] = set()
        for p in field_pieces:
            xz = by.get(p["ship_id"])
            if xz and xz not in used:
                p["x"], p["z"] = xz
                used.add(xz)
            else:
                cells = [c for c in legal_field_cells(board) if c not in used]
                if cells:
                    p["x"], p["z"] = cells[0]
                    used.add(cells[0])
    return pieces


def fleet_ids_from_pieces(pieces: list[dict[str, Any]], ships: dict[str, Any] | None = None) -> list[str]:
    field = [p["ship_id"] for p in pieces if p["slot"] == "field"]
    ships = ships or {}
    has_cyno = any(is_covert_cyno(ships.get(str(p["ship_id"]))) for p in pieces if p["slot"] == "field")
    if has_cyno:
        jumped = [
            p["ship_id"]
            for p in pieces
            if p["slot"] == "hangar" and is_cyno_flagship(ships.get(str(p["ship_id"])))
        ]
        return field + jumped
    return field or [p["ship_id"] for p in pieces]
