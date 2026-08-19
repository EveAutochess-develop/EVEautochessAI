"""Scripted miner_flag FSM opponent: shop rules, deploy, fight_units, farm feedback."""

from __future__ import annotations

import random
from eveac_ai.match20 import apply_seat_checkpoint, fight_units
from eveac_ai.seat_prep import new_seat_board, prepare_turn
from eveac_ai.scripted_fsm import (
    format_fsm_feedback,
    fsm_win_rate,
    pick_scripted_seat_ids,
    prepare_miner_flag,
    league_fsm_stats,
    empty_fsm_stats,
)


class _C:
    ships = {
        "10": {"id": 10, "name": "富豪级", "cost": 2, "ship_group": "frigate"},
        "43": {"id": 43, "name": "守护级", "cost": 7, "is_logistic": True, "ship_group": "battlecruiser"},
        "101": {
            "id": 101,
            "name": "主宰级隐匿型",
            "cost": 5,
            "capital_role": "covert_cyno",
            "ship_group": "cruiser",
        },
        "111": {
            "id": 111,
            "name": "神示级",
            "cost": 24,
            "requires_cyno_entry": True,
            "capital_role": "dreadnought",
            "ship_group": "dreadnought",
        },
        "135": {
            "id": 135,
            "name": "回旋者级",
            "cost": 3,
            "is_mining_ship": True,
            "ship_group": "mining_barge",
        },
        "136": {
            "id": 136,
            "name": "海豚级",
            "cost": 5,
            "is_mining_ship": True,
            "ship_group": "mining_barge",
        },
        "138": {
            "id": 138,
            "name": "长须鲸级",
            "cost": 22,
            "is_mining_ship": True,
            "capital_role": "capital_industrial",
            "ship_group": "capital_industrial",
            "requires_cyno_entry": False,
        },
        "812": {
            "id": 812,
            "name": "FAX",
            "cost": 24,
            "is_logistic": True,
            "requires_cyno_entry": True,
            "capital_role": "force_auxiliary",
            "ship_group": "force_auxiliary",
        },
    }
    economy = {
        "buy_exp_gold_cost": 4,
        "buy_exp_amount": 4,
        "player_level_cap": 20,
        "refresh_cost": 2,
        "initial_level_exp_demand": 4,
        "level_exp_demand_increment": 8,
    }
    board = {"field_width": 8, "field_height": 4, "hangar_width": 8, "field_odd_row_extra": 0}


def _board(**kw):
    b = new_seat_board()
    b.update(kw)
    return b


def test_pick_scripted_nullsec_two_of_twenty():
    for seed in range(20):
        ids = pick_scripted_seat_ids(20, 0.10, random.Random(seed), security_mode="nullsec")
        assert len(ids) == 2
        assert ids <= set(range(20))


def test_pick_scripted_lowsec_ten_percent_of_tables():
    hits = 0
    n = 2000
    for seed in range(n):
        ids = pick_scripted_seat_ids(2, 0.10, random.Random(seed), security_mode="lowsec")
        assert len(ids) <= 1
        if ids:
            hits += 1
            assert ids <= {0, 1}
    rate = hits / n
    assert 0.06 < rate < 0.14


def test_early_combat_shop_buys_nothing():
    board = _board(gold=5, level=1)
    combat = ["10"] * 6
    prepare_miner_flag(
        _C(),
        {},
        "amarr",
        board,
        random.Random(0),
        _C.board,
        shop_ships=list(combat),
        roll_ships=lambda: list(combat),
    )
    assert board["pieces"] == []
    assert int(board["gold"]) < 5


def test_miner_shop_buys_and_merges_star2():
    board = _board(gold=12, level=3)
    miners = ["135"] * 6
    prepare_miner_flag(
        _C(),
        {},
        "amarr",
        board,
        random.Random(1),
        _C.board,
        shop_ships=list(miners),
        roll_ships=lambda: list(miners),
    )
    owned = [p for p in board["pieces"] if str(p["ship_id"]) == "135"]
    assert owned
    assert any(int(p.get("star") or 1) >= 2 for p in owned)
    assert not any(str(p["ship_id"]) == "10" for p in board["pieces"])


def test_lv15_rorqual_cyno_field_flag_hangar_fight_units():
    board = _board(gold=100, level=15)
    offer = ["138", "101", "111", "10", "10", "10"]
    prepare_miner_flag(
        _C(),
        {},
        "amarr",
        board,
        random.Random(2),
        _C.board,
        shop_ships=list(offer),
        roll_ships=lambda: list(offer),
    )
    by = {str(p["ship_id"]): p for p in board["pieces"]}
    assert "138" in by and by["138"]["slot"] == "field"
    assert "101" in by and by["101"]["slot"] == "field"
    assert "111" in by and by["111"]["slot"] == "hangar"
    assert not any(str(p["ship_id"]) == "10" for p in board["pieces"])
    units = fight_units(board["pieces"], _C.ships)
    hold = [u for u in units if u.get("cyno_hold")]
    assert any(u["ship_id"] == "111" for u in hold)
    assert any(u["ship_id"] == "101" and not u.get("cyno_hold") for u in units)


def test_prepare_turn_scripted_skips_nets():
    class Boom:
        def prepare_seat(self, *a, **k):
            raise AssertionError("nets must not run for scripted seats")

    board = _board(gold=5, level=1)
    combat = ["10"] * 6
    tr = prepare_turn(
        _C(),
        {},
        "amarr",
        board,
        random.Random(0),
        _C.board,
        Boom(),
        scripted="miner_flag",
    )
    # shop still rolled by FSM unless we inject; empty combat-only via real roll may buy miners.
    # Force inject by calling with scripted after setting shop is not available; just check trace tag
    # when shop happens to contain miners. The Boom assertion is the contract.
    assert tr and tr.get("scripted") == "miner_flag"


def test_checkpoint_roundtrips_scripted():
    seat = {
        "seat_id": 0,
        "titan": "amarr",
        "board": new_seat_board(),
        "gold": 5,
        "alive": True,
        "titan_hp": 300,
        "wins": 0,
        "losses": 0,
        "fleet": [],
        "genome": {},
        "memory": {},
        "frozen": False,
        "scripted": "",
    }
    apply_seat_checkpoint(
        seat,
        {
            "titan": "gallente",
            "titan_hp": 80,
            "wins": 1,
            "losses": 2,
            "alive": True,
            "gold": 20,
            "board": {"gold": 20, "level": 8, "shop": ["135"], "pieces": []},
            "frozen": True,
            "scripted": "miner_flag",
        },
    )
    assert seat["scripted"] == "miner_flag"
    assert seat["frozen"] is True


def test_fsm_match_win_rate_feedback():
    stats = {"kind": "miner_flag", "match_wins": 1, "match_n": 5, "flag_jump": 8, "flag_hangar": 11}
    assert abs(fsm_win_rate(stats) - 0.2) < 1e-9
    assert format_fsm_feedback(stats, gen=12) == "FSM wr=0.200"


def test_league_fsm_stats_uses_finalize_winner():
    league = type(
        "L",
        (),
        {
            "seats": [
                {"seat_id": 0, "scripted": "miner_flag"},
                {"seat_id": 1, "scripted": ""},
            ],
            "stats": {"fsm_flag_jump": 3, "fsm_flag_hangar": 4},
        },
    )()
    out_win = {"ranked": [{"seat_id": 0}, {"seat_id": 1}]}
    st = league_fsm_stats(league, out_win)
    assert st["match_wins"] == 1 and st["match_n"] == 1
    out_lose = {"ranked": [{"seat_id": 1}, {"seat_id": 0}]}
    st2 = league_fsm_stats(league, out_lose)
    assert st2["match_wins"] == 0 and st2["match_n"] == 1


def test_buy_logi_skips_fax():
    board = _board(gold=40, level=8)
    shop = ["812", "10", "10", "10", "10", "10"]
    prepare_miner_flag(
        _C(),
        {},
        "amarr",
        board,
        random.Random(3),
        _C.board,
        shop_ships=list(shop),
        roll_ships=lambda: list(shop),
    )
    assert not any(str(p["ship_id"]) == "812" for p in board["pieces"])
    assert not any(str(p["ship_id"]) == "10" for p in board["pieces"])
