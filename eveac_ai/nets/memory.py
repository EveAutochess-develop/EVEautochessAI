"""Per-seat match memory: named boards, alive roster, opponent priors. No live shop peek."""

from __future__ import annotations

from typing import Any

from eveac_ai import TITAN_IDS
from eveac_ai.content import titan_max_hp, remaining_pvp_losses
from eveac_ai.prepare import is_cyno_flagship, is_covert_cyno

N_SEATS = 20
ROUND_DIM = 16
KEEP_ROUNDS = 8
MG_DIM = 20 + 20 + 20 + 5 + KEEP_ROUNDS * ROUND_DIM + 8


def _cost(content: Any, pieces: list[dict], slot: str = "field") -> float:
    s = 0.0
    for p in pieces:
        if p.get("slot") != slot:
            continue
        s += float((content.ships.get(str(p["ship_id"])) or {}).get("cost") or 0)
    return s


def field_brief(content: Any, pieces: list[dict]) -> list[float]:
    field = [p for p in pieces if p.get("slot") == "field"]
    stars = sum(int(p.get("star") or 1) for p in field)
    cyno = 1.0 if any(is_covert_cyno(content.ships.get(str(p["ship_id"]))) for p in field) else 0.0
    flag = 1.0 if any(is_cyno_flagship(content.ships.get(str(p["ship_id"]))) for p in pieces) else 0.0
    return [len(field) / 10.0, _cost(content, pieces) / 50.0, stars / 15.0, cyno, flag]


def round_summary(
    content: Any,
    *,
    me_id: int,
    opp_id: int | None,
    me_pieces: list[dict],
    opp_pieces: list[dict] | None,
    won: float,
    dmg_self: float,
    dmg_enemy: float,
    bye: bool,
) -> list[float]:
    opp = opp_pieces or []
    row = [
        me_id / float(N_SEATS),
        (-1.0 if opp_id is None else opp_id / float(N_SEATS)),
        *field_brief(content, me_pieces),
        *field_brief(content, opp),
        1.0 if won else 0.0,
        1.0 if bye else 0.0,
        min(1.0, dmg_self / 4000.0),
        min(1.0, dmg_enemy / 4000.0),
    ]
    return (row + [0.0] * ROUND_DIM)[:ROUND_DIM]


def new_memory() -> dict[str, Any]:
    return {"rounds": [], "fought": set(), "seen_field_cost": [0.0] * N_SEATS}


def hydrate_memory(raw: object) -> dict[str, Any]:
    """Rebuild runtime memory after JSON (sets became lists or str(set))."""
    mem = new_memory()
    if not isinstance(raw, dict):
        return mem
    mem["rounds"] = list(raw.get("rounds") or [])
    fought = raw.get("fought")
    ids: set[int] = set()
    if isinstance(fought, str):
        blob = fought.strip().lstrip("{").rstrip("}")
        parts = [p.strip() for p in blob.split(",") if p.strip()] if blob else []
        for part in parts:
            try:
                ids.add(int(part))
            except ValueError:
                continue
    elif isinstance(fought, (list, tuple, set, frozenset)):
        for x in fought:
            try:
                ids.add(int(x))
            except (TypeError, ValueError):
                continue
    mem["fought"] = ids
    sc = raw.get("seen_field_cost") or [0.0] * N_SEATS
    if isinstance(sc, list):
        out = [float(x) for x in sc[:N_SEATS]]
        if len(out) < N_SEATS:
            out.extend([0.0] * (N_SEATS - len(out)))
        mem["seen_field_cost"] = out
    return mem


def match_global_obs(
    *,
    seats: list[dict],
    viewer: dict,
    rnd: int,
    n_seats: int,
    security_mode: str = "nullsec",
) -> list[float]:
    mem = hydrate_memory(viewer.get("memory"))
    viewer["memory"] = mem
    alive = [1.0 if (i < len(seats) and seats[i]["alive"]) else 0.0 for i in range(N_SEATS)]
    last_c = []
    sc = mem.get("seen_field_cost") or [0.0] * N_SEATS
    for i in range(N_SEATS):
        last_c.append(float(sc[i]) / 50.0 if i < len(sc) else 0.0)
    fought_set = mem.get("fought") if isinstance(mem.get("fought"), set) else set()
    fought = [1.0 if i in fought_set else 0.0 for i in range(N_SEATS)]
    cen = [0.0] * 5
    ids = list(TITAN_IDS)
    for s in seats:
        if not s.get("alive"):
            continue
        t = str(s.get("titan") or "")
        if t in ids:
            cen[ids.index(t)] += 1.0
    ssum = sum(cen) or 1.0
    cen = [x / ssum for x in cen]
    hist: list[float] = []
    for r in list(mem.get("rounds") or [])[-KEEP_ROUNDS:]:
        hist.extend(r)
    hist += [0.0] * (KEEP_ROUNDS * ROUND_DIM - len(hist))
    alive_n = sum(alive)
    known = [last_c[i] for i in range(N_SEATS) if alive[i] > 0.5]
    expect = (sum(known) / len(known)) if known else 0.0
    mode = "lowsec" if str(security_mode) == "lowsec" else "nullsec"
    extra = [
        rnd / 20.0,
        alive_n / max(1.0, float(n_seats or N_SEATS)),
        expect,
        remaining_pvp_losses(float(viewer.get("titan_hp") or 0)) / max(1, remaining_pvp_losses(titan_max_hp())),
        1.0 if mode == "nullsec" else 0.0,
        1.0 if mode == "lowsec" else 0.0,
        0.0,
        1.0,
    ]
    return (alive + last_c + fought + cen + hist + extra)[:MG_DIM]


def note_fight(mem: dict[str, Any], opp_id: int, opp_field_cost: float) -> None:
    if not isinstance(mem.get("fought"), set):
        mem["fought"] = hydrate_memory(mem)["fought"]
    mem["fought"].add(int(opp_id))
    sc = mem.setdefault("seen_field_cost", [0.0] * N_SEATS)
    if 0 <= opp_id < len(sc):
        sc[opp_id] = float(opp_field_cost)
