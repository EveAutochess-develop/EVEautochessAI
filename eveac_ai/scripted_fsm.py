"""Frozen miner→logi→cyno/rorqual→flagship sparring partner (farm seats only)."""

from __future__ import annotations

import random
from typing import Any, Callable

from eveac_ai.economy import xp_demand
from eveac_ai.prepare import (
    is_cyno_flagship,
    is_covert_cyno,
    is_mining_hull,
    is_shop_combat_hull,
    legal_field_cells,
    _place_relative,
)
from eveac_ai.seat_prep import _pop_cap, try_merge
from eveac_ai.shop import roll_ship_shop

SCRIPTED_KIND = "miner_flag"
RORQUAL_ID = "138"
PORPOISE_ID = "136"
MAX_STEPS = 48
INTEREST_GOLD = 10

RollShips = Callable[[], list[str]]


def pick_scripted_seat_ids(
    n_seats: int,
    frac: float,
    rng: random.Random,
    *,
    security_mode: str,
) -> set[int]:
    """Nullsec: round(n*frac) seats. Lowsec (2 seats): frac of tables get 1 seat."""
    n_seats = int(n_seats)
    frac = max(0.0, min(1.0, float(frac)))
    if n_seats <= 0 or frac <= 0:
        return set()
    mode = str(security_mode or "nullsec")
    if mode == "lowsec" or n_seats <= 2:
        if rng.random() < frac:
            return {int(rng.randrange(n_seats))}
        return set()
    k = int(round(n_seats * frac))
    k = max(0, min(k, n_seats))
    if k <= 0:
        return set()
    return set(rng.sample(range(n_seats), k))


def empty_fsm_stats(*, kind: str = SCRIPTED_KIND) -> dict[str, Any]:
    return {
        "kind": str(kind or SCRIPTED_KIND),
        "seats": 0,
        "match_wins": 0,
        "match_n": 0,
        "flag_jump": 0,
        "flag_hangar": 0,
    }


def add_fsm_stats(dst: dict[str, Any], src: dict[str, Any] | None) -> dict[str, Any]:
    if not src:
        return dst
    for k in ("seats", "match_wins", "match_n", "flag_jump", "flag_hangar"):
        dst[k] = int(dst.get(k) or 0) + int(src.get(k) or 0)
    if src.get("kind"):
        dst["kind"] = str(src.get("kind"))
    return dst


def fsm_win_rate(stats: dict[str, Any] | None) -> float | None:
    """Share of FSM seats that ranked #1 at league finalize (whole match, not per-round PVP)."""
    if not stats:
        return None
    n = int(stats.get("match_n") or 0)
    if n <= 0:
        return None
    return int(stats.get("match_wins") or 0) / n


def format_fsm_feedback(
    stats: dict[str, Any] | None,
    *,
    gen: int,
    lifetime: dict[str, Any] | None = None,
) -> str:
    st = stats or empty_fsm_stats()
    wr = fsm_win_rate(st)
    wr_s = f"{wr:.3f}" if wr is not None else "na"
    return f"FSM wr={wr_s}"


def league_fsm_stats(league: Any, out: dict | None = None) -> dict[str, Any]:
    st = empty_fsm_stats()
    ranked = (out or {}).get("ranked") or []
    winner_id = ranked[0]["seat_id"] if ranked else None
    for s in getattr(league, "seats", None) or []:
        if not s.get("scripted"):
            continue
        st["seats"] += 1
        st["match_n"] += 1
        if winner_id is not None and int(s["seat_id"]) == int(winner_id):
            st["match_wins"] += 1
    raw = getattr(league, "stats", None) or {}
    st["flag_jump"] = int(raw.get("fsm_flag_jump") or 0)
    st["flag_hangar"] = int(raw.get("fsm_flag_hangar") or 0)
    return st


def _hull(content: Any, sid: Any) -> dict[str, Any]:
    return content.ships.get(str(sid)) or {}


def _is_field_logi(hull: dict[str, Any]) -> bool:
    return bool(hull.get("is_logistic")) and not is_cyno_flagship(hull)


def _is_rorqual(hull: dict[str, Any], sid: str) -> bool:
    if str(sid) == RORQUAL_ID:
        return True
    return str(hull.get("ship_group") or "").lower() == "capital_industrial" or str(
        hull.get("capital_role") or ""
    ).lower() == "capital_industrial"


def _is_early_miner(hull: dict[str, Any], sid: str) -> bool:
    if is_cyno_flagship(hull) or is_covert_cyno(hull) or _is_rorqual(hull, sid):
        return False
    return bool(is_mining_hull(hull) or hull.get("is_mining_ship"))


def _phase(content: Any, board: dict[str, Any]) -> str:
    lv = int(board.get("level") or 1)
    gold = int(board.get("gold") or 0)
    pieces = list(board.get("pieces") or [])
    has_flag = any(is_cyno_flagship(_hull(content, p.get("ship_id"))) for p in pieces)
    has_cyno = any(is_covert_cyno(_hull(content, p.get("ship_id"))) for p in pieces)
    if lv >= 15:
        if has_flag and has_cyno:
            return "flag"
        return "wall"
    if lv >= 8 and gold >= INTEREST_GOLD:
        return "logi"
    return "mine"


def _buy_score(phase: str, hull: dict[str, Any], sid: str) -> int:
    if not is_shop_combat_hull(hull):
        return 0
    if phase == "mine":
        return 10 if _is_early_miner(hull, sid) else 0
    if phase == "logi":
        if _is_field_logi(hull):
            return 30
        if _is_early_miner(hull, sid):
            return 10
        return 0
    if phase == "wall":
        if is_covert_cyno(hull):
            return 50
        if _is_rorqual(hull, sid):
            return 40
        if str(sid) == PORPOISE_ID:
            return 30
        if is_cyno_flagship(hull):
            return 25
        if _is_early_miner(hull, sid):
            return 20
        if _is_field_logi(hull):
            return 10
        return 0
    if phase == "flag":
        if is_cyno_flagship(hull):
            return 100
        if is_covert_cyno(hull):
            return 40
        if _is_rorqual(hull, sid):
            return 30
        if str(sid) == PORPOISE_ID:
            return 20
        if _is_field_logi(hull):
            return 10
        if _is_early_miner(hull, sid):
            return 5
        return 0
    return 0


def _field_score(content: Any, piece: dict[str, Any], pieces: list[dict[str, Any]]) -> int:
    sid = str(piece.get("ship_id"))
    hull = _hull(content, sid)
    if is_cyno_flagship(hull):
        return -1
    has_flag = any(is_cyno_flagship(_hull(content, p.get("ship_id"))) for p in pieces)
    if is_covert_cyno(hull):
        return 1000 if has_flag else 90
    if _is_rorqual(hull, sid):
        return 80
    if _is_field_logi(hull):
        return 60
    if str(sid) == PORPOISE_ID:
        return 50
    if _is_early_miner(hull, sid) or is_mining_hull(hull):
        return 40
    return 10


def deploy_miner_flag(content: Any, genome: dict, board_state: dict, board_desc: dict) -> None:
    pieces = board_state["pieces"]
    cap = _pop_cap(board_state)
    ranked = sorted(
        (p for p in pieces if _field_score(content, p, pieces) >= 0),
        key=lambda p: (-_field_score(content, p, pieces), int(p.get("token") or 0)),
    )
    chosen = set(id(p) for p in ranked[:cap])
    for p in pieces:
        if is_cyno_flagship(_hull(content, p.get("ship_id"))):
            p["slot"] = "hangar"
        elif id(p) in chosen:
            p["slot"] = "field"
        else:
            p["slot"] = "hangar"
    stance = (genome or {}).get("stance") or {}
    fps = [p for p in pieces if p.get("slot") == "field"]
    laid = _place_relative([p["ship_id"] for p in fps], content, board_desc, stance)
    used: set[tuple[int, int]] = set()
    by: dict[str, list[tuple[int, int]]] = {}
    for sid, x, z in laid:
        by.setdefault(str(sid), []).append((x, z))
    for p in fps:
        opts = by.get(str(p["ship_id"])) or []
        xz = None
        while opts:
            xz = opts.pop(0)
            if xz not in used:
                break
            xz = None
        if xz is None:
            cells = [c for c in legal_field_cells(board_desc) if c not in used]
            xz = cells[0] if cells else (0, 0)
        p["x"], p["z"] = xz
        used.add(xz)


def _buy_xp(content: Any, board: dict[str, Any]) -> bool:
    econ = content.economy
    cost = int(econ.get("buy_exp_gold_cost") or 4)
    cap = int(econ.get("player_level_cap") or 20)
    if int(board.get("gold") or 0) < cost:
        return False
    if int(board.get("level") or 1) >= cap:
        return False
    board["gold"] = int(board["gold"]) - cost
    board["xp"] = int(board.get("xp") or 0) + int(econ.get("buy_exp_amount") or 4)
    need = xp_demand(econ, int(board["level"]))
    if int(board["xp"]) >= need:
        board["xp"] = int(board["xp"]) - need
        board["level"] = int(board["level"]) + 1
    return True


def prepare_miner_flag(
    content: Any,
    genome: dict,
    titan: str,
    board_state: dict,
    rng: random.Random,
    board_desc: dict,
    *,
    shop_ships: list[str] | None = None,
    roll_ships: RollShips | None = None,
) -> dict[str, Any]:
    econ = content.economy
    roll = roll_ships or (
        lambda: roll_ship_shop(content, level=int(board_state.get("level") or 1), titan=titan, rng=rng)
    )
    if shop_ships is not None:
        board_state["shop_ships"] = list(shop_ships)
    else:
        board_state["shop_ships"] = roll()
    board_state.setdefault("shop_equips", [])
    xp_buys = 0
    last_phase = _phase(content, board_state)
    steps = 0
    while steps < MAX_STEPS:
        steps += 1
        last_phase = _phase(content, board_state)
        gold = int(board_state.get("gold") or 0)
        cap = _pop_cap(board_state)
        fieldable = sum(
            1 for p in board_state["pieces"] if not is_cyno_flagship(_hull(content, p.get("ship_id")))
        )
        if last_phase in ("mine", "logi") and fieldable >= cap and _buy_xp(content, board_state):
            xp_buys += 1
            continue
        best_i, best_s = -1, 0
        for i, sid in enumerate(board_state.get("shop_ships") or []):
            if not sid:
                continue
            hull = _hull(content, sid)
            cost = int(float(hull.get("cost") or 99))
            if cost > gold:
                continue
            score = _buy_score(last_phase, hull, str(sid))
            if score > best_s:
                best_s, best_i = score, i
        if best_i >= 0:
            sid = str(board_state["shop_ships"][best_i])
            hull = _hull(content, sid)
            cost = int(float(hull.get("cost") or 0))
            board_state["gold"] = gold - cost
            tok = int(board_state.get("token") or 1)
            board_state["token"] = tok + 1
            board_state["pieces"].append(
                {"token": tok, "ship_id": sid, "star": 1, "equips": [], "slot": "hangar", "x": 0, "z": 0}
            )
            board_state["shop_ships"][best_i] = ""
            try_merge(board_state["pieces"])
            continue
        refresh = int(econ.get("refresh_cost") or 2)
        if gold >= refresh:
            wanted = False
            for sid in board_state.get("shop_ships") or []:
                if not sid:
                    continue
                hull = _hull(content, sid)
                cost = int(float(hull.get("cost") or 99))
                if cost <= gold - refresh and _buy_score(last_phase, hull, str(sid)) > 0:
                    wanted = True
                    break
            if not wanted:
                board_state["gold"] = gold - refresh
                board_state["shop_ships"] = roll()
                continue
        break
    deploy_miner_flag(content, genome or {}, board_state, board_desc)
    pieces = board_state["pieces"]
    cap = _pop_cap(board_state)
    field_n = sum(1 for p in pieces if p.get("slot") == "field")
    return {
        "bought_xp": xp_buys > 0,
        "field_full": field_n >= cap,
        "skipped_xp_when_full": False,
        "scripted": SCRIPTED_KIND,
        "phase": last_phase,
    }
