"""Round gold / XP / sell prices from economy.json."""

from __future__ import annotations

from typing import Any
import math


def income_base(econ: dict[str, Any], round_i: int) -> int:
    by = econ.get("base_gold_income_by_round") or [2, 3, 4]
    if round_i < len(by):
        return int(by[round_i])
    return int(econ.get("base_gold_income") or 5)


def interest_of(econ: dict[str, Any], gold: int) -> int:
    """MatchController._compute_round_income_parts: min(floor(gold/10), cap)."""
    div = int(econ.get("interest_divisor") or 10)
    cap = int(econ.get("interest_cap") or 5)
    v = int(gold) // max(div, 1)
    if econ.get("interest_capped", True):
        v = min(cap, v)
    return int(v)


def _is_porpoise(hull: dict[str, Any], ship_id: str) -> bool:
    if str(ship_id) == "136":
        return True
    return "mining_command" in [str(x) for x in (hull.get("fetter_ids") or [])]


def mining_gold_from_survivors(content: Any, units: list[dict[str, Any]]) -> int:
    """MINING §4 / MatchController._mining_gold_for_team: Field survivors only.

    Porpoise itself is not multiplied. Other sources floor(starred × cmd_mul).
    Extra Porpoise beyond 1 adds +1 percentage point. Rorqual excavators: if the
    kernel did not spawn them, pay table count only while the mother survived.
    """
    live = [u for u in units if u.get("survived", True)]
    porpoise_n = 0
    for u in live:
        if u.get("is_unmanned"):
            continue
        hull = content.ships.get(str(u.get("ship_id"))) or {}
        if _is_porpoise(hull, str(u.get("ship_id"))):
            porpoise_n += 1
    has_command = porpoise_n > 0
    cmd_mul = 1.0
    if has_command:
        base_pct = 20.0
        mc = (content.fetters or {}).get("mining_command") or {}
        for e in mc.get("effects") or []:
            if isinstance(e, dict) and str(e.get("effect_type") or "") == "MiningGoldBonus":
                base_pct = float(e.get("value") or 20.0)
                break
        cmd_mul = 1.0 + (base_pct + float(max(0, porpoise_n - 1))) / 100.0
    total = 0
    for u in live:
        hull = content.ships.get(str(u.get("ship_id"))) or {}
        base_g = int(float(hull.get("mining_gold_per_round") or 0))
        star = max(1, int(u.get("star") or 1))
        if u.get("unmanned_kind") == "mining_excavator":
            if base_g <= 0:
                base_g = 25
            starred = base_g * star
            total += int(math.floor(starred * cmd_mul)) if has_command else starred
            continue
        if base_g <= 0:
            continue
        starred = base_g * star
        is_p = (not u.get("is_unmanned")) and _is_porpoise(hull, str(u.get("ship_id")))
        if has_command and not is_p:
            total += int(math.floor(float(starred) * cmd_mul))
        else:
            total += starred
        drones = int(float(hull.get("mining_drone_count") or 0))
        if drones and not any(x.get("unmanned_kind") == "mining_excavator" for x in units):
            dpay = 25 * star * drones
            total += int(math.floor(float(dpay) * cmd_mul)) if has_command else dpay
    return int(total)


def mining_gold_of(content: Any, pieces: list[dict], *, all_survived: bool = True) -> int:
    units = []
    for p in pieces:
        if p.get("slot") != "field":
            continue
        hull = content.ships.get(str(p["ship_id"])) or {}
        units.append(
            {
                "ship_id": str(p["ship_id"]),
                "star": int(p.get("star") or 1),
                "survived": all_survived,
                "is_unmanned": bool(hull.get("is_unmanned")),
            }
        )
    return mining_gold_from_survivors(content, units)


def round_income_pre(
    econ: dict[str, Any],
    *,
    gold_ref: int,
    round_i: int,
    won: bool,
    win_streak: int,
    mining_g: int,
) -> dict[str, int]:
    """Combat-end packet before loss_comp. Kills are not in this packet."""
    base = income_base(econ, round_i)
    interest = interest_of(econ, gold_ref)
    win_g = int(econ.get("win_gold") or 1) if won else 0
    streak_g = streak_gold(econ, win_streak, won)
    mining_g = int(mining_g)
    income = base + interest + win_g + streak_g + mining_g
    return {
        "base": base,
        "interest": interest,
        "win": win_g,
        "streak": streak_g,
        "mining": mining_g,
        "income": income,
    }


def grant_exp(econ: dict[str, Any], board: dict, amount: int) -> None:
    if amount <= 0:
        return
    board["xp"] = int(board.get("xp") or 0) + int(amount)
    cap = int(econ.get("player_level_cap") or 20)
    while int(board.get("level") or 1) < cap:
        need = xp_demand(econ, int(board["level"]))
        if int(board["xp"]) < need:
            break
        board["xp"] -= need
        board["level"] += 1



def streak_gold(econ: dict[str, Any], wins_in_a_row: int, won: bool) -> int:
    if not won:
        return 0
    table = econ.get("streak_gold") or {}
    best = 0
    for k, v in table.items():
        if wins_in_a_row >= int(k):
            best = max(best, int(v))
    return best


def sell_price(econ: dict[str, Any], cost: float) -> int:
    disc = int(econ.get("sell_price_discount") or 3)
    mn = int(econ.get("sell_price_min") or 1)
    return max(mn, int(cost) - disc)


def loss_comp(
    econ: dict[str, Any],
    *,
    loss_streak: int,
    winner_income: int,
    winner_field_cost: int,
    titan_pvp: dict | None = None,
    loser_core: int = 0,
) -> int:
    start = float(econ.get("loss_comp_rate_start") or 0.10)
    step = float(econ.get("loss_comp_rate_step") or 0.20)
    cap = float(econ.get("loss_comp_rate_cap") or 0.70)
    n = max(1, int(loss_streak))
    rate = min(cap, start + step * (n - 1))
    raw = math.ceil((winner_income + winner_field_cost) * rate)
    tp = titan_pvp or {}
    cap_mul = float(tp.get("loss_comp_vs_winner_cap") or econ.get("loss_comp_vs_winner_cap") or 0.75)
    less_n = int(tp.get("loss_comp_vs_winner_less") or econ.get("loss_comp_vs_winner_less") or 60)
    cap_pct = math.floor(winner_income * cap_mul)
    cap_flat = max(0, winner_income - less_n)
    ceiling = max(cap_pct, cap_flat)
    total = int(loser_core) + int(raw)
    if winner_income and ceiling > 0:
        total = min(total, int(ceiling))
    return max(0, total - int(loser_core))


def xp_demand(econ: dict[str, Any], level: int) -> int:
    init = int(econ.get("initial_level_exp_demand") or 4)
    inc = int(econ.get("level_exp_demand_increment") or 8)
    return init + inc * max(0, level - 1)
