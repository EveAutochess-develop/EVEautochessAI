"""Nullsec PVE schedule and sleeper/freighter lock. Farm fights for real (not ai_player instant win)."""

from __future__ import annotations

import zlib
from typing import Any

from eveac_ai.prepare import legal_field_cells

TASK_ELIMINATE = "pve_eliminate"
TASK_SALVAGE = "pve_salvage"


def is_pvp_round(round_r: int) -> bool:
    """1-based R. MULTIPLAYER_MATCH_FLOW §3 / NullsecPveDirector.is_pvp_round."""
    if round_r <= 4:
        return round_r == 3
    return round_r % 2 == 0


def roll_pve_task(rng: Any, round_r: int, match_seed: int | None = None) -> str:
    """Same round, all seats: crc32(match_seed:pve_task:round) like Godot stream."""
    seed = match_seed
    if seed is None:
        seed = int(getattr(rng, "match_seed", 0) or 0)
    h = zlib.crc32(f"{int(seed)}:pve_task:{int(round_r)}".encode("utf-8")) & 0x7FFFFFFF
    return TASK_SALVAGE if (h % 2) == 1 else TASK_ELIMINATE


def _tags(hull: dict) -> list[str]:
    return [str(t).lower() for t in (hull.get("tags") or [])]


def sleeper_pool(content: Any, level: int) -> list[str]:
    unlocks = (content.economy or {}).get("shop_unlock_level_by_group") or {}
    out: list[str] = []
    for sid, hull in (content.ships or {}).items():
        tags = _tags(hull)
        if "sleeper" not in tags and "pve_creep" not in tags:
            continue
        group = str(hull.get("ship_group") or "frigate")
        need = int(unlocks.get(group, 1) or 1)
        if int(level) >= need:
            out.append(str(sid))
    return out


def _creep_cost(hull: dict) -> int:
    cost = int(float(hull.get("cost") or 0) or 0)
    if cost <= 0:
        cost = 1
    if hull.get("is_mining_ship"):
        cost = max(1, (cost + 1) // 2)
    return cost


def lock_creeps(
    content: Any,
    rng: Any,
    *,
    gold: int,
    level: int,
    pop_limit: int,
    field_value: int,
) -> list[dict[str, Any]]:
    """PveCreepAi.lock_from_player_state: budget = floor(gold/2)+field cost; cap = pop*1.5."""
    cap = max(1, int(float(pop_limit) * 1.5))
    budget = max(0, int(gold) // 2) + max(0, int(field_value))
    pool = sleeper_pool(content, level)
    cells = legal_field_cells(content.board)
    if not pool:
        return []
    roster: list[dict[str, Any]] = []
    spent = 0
    guard = 0
    while spent < budget and len(roster) < cap and guard < 64:
        guard += 1
        sid = pool[rng.randrange(len(pool))]
        hull = content.ships.get(sid) or {}
        cost = _creep_cost(hull)
        if spent + cost > budget and roster:
            break
        spent += cost
        cell_i = rng.randrange(32)
        x, z = cells[cell_i % len(cells)] if cells else (0, 0)
        roster.append({"ship_id": sid, "x": int(x), "z": int(z), "star": 1, "equips": [], "slot": "field"})
    if not roster and pool:
        sid = pool[0]
        x, z = cells[0] if cells else (0, 0)
        roster.append({"ship_id": sid, "x": int(x), "z": int(z), "star": 1, "equips": [], "slot": "field"})
    return roster


def _ship_race(hull: dict) -> str:
    race = str(hull.get("race") or "").strip().lower()
    if race:
        return race
    for t in list(hull.get("fetter_ids") or []) + _tags(hull):
        k = str(t).lower()
        if k in ("amarr", "caldari", "gallente", "minmatar", "angel"):
            return k
    return ""


def freighter_pool(content: Any, exclude_race: str = "") -> list[str]:
    exclude = str(exclude_race or "").strip().lower()
    out: list[str] = []
    for sid, hull in (content.ships or {}).items():
        tags = _tags(hull)
        group = str(hull.get("ship_group") or "").lower()
        if not ("freighter" in tags or "pve_salvage" in tags or group == "freighter"):
            continue
        if exclude and _ship_race(hull) == exclude:
            continue
        out.append(str(sid))
    if not out and exclude:
        return freighter_pool(content, "")
    return out or ["211"]


def salvage_freighter_unit(content: Any, rng: Any, titan: str) -> dict[str, Any]:
    pool = freighter_pool(content, str(titan or "").replace("titan_", ""))
    sid = pool[rng.randrange(len(pool))]
    cells = legal_field_cells(content.board)
    front = [c for c in cells if c[1] == 0] or cells
    x, z = front[len(front) // 2] if front else (0, 0)
    return {
        "ship_id": str(sid),
        "x": int(x),
        "z": int(z),
        "star": 1,
        "equips": [],
        "slot": "field",
        "pve_freighter": True,
    }


def pve_success(*, task: str, row_player: dict, row_creep: dict, freighter_id: str) -> bool:
    if task == TASK_SALVAGE:
        fid = str(freighter_id or "")
        for s in row_player.get("ships") or []:
            if str(s.get("ship_id") or "") == fid and s.get("survived"):
                return True
        return False
    manned_live = any(
        (not s.get("is_unmanned")) and s.get("survived") for s in (row_creep.get("ships") or [])
    )
    return not manned_live
