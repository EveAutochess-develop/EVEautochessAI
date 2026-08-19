import random

from eveac_ai.prepare import fleet_ids_from_pieces, is_shop_combat_hull, pick_equips, prepare_pieces
from eveac_ai.priors import pick_fleet


class _C:
    ships = {
        "10": {"name": "富豪级", "cost": 2},
        "11": {"name": "小鹰级", "cost": 3},
        "12": {"name": "秃鹫级", "cost": 5},
        "101": {"name": "主宰级隐匿型", "cost": 5, "capital_role": "covert_cyno"},
        "111": {"name": "神使级", "cost": 24, "requires_cyno_entry": True, "capital_role": "dreadnought"},
        "201": {"name": "泰坦", "cost": 99, "capital_role": "titan", "ship_group": "titan"},
    }
    equip_ids: list[str] = []
    equip_meta: dict = {}


def test_cyno_and_flagship_are_shop_hulls():
    assert is_shop_combat_hull(_C.ships["101"])
    assert is_shop_combat_hull(_C.ships["111"])
    assert not is_shop_combat_hull(_C.ships["201"])
    assert is_shop_combat_hull({"name": "回旋者级", "cost": 3, "is_mining_ship": True, "ship_group": "mining_barge"})
    assert is_shop_combat_hull({"name": "长须鲸级", "cost": 22, "is_mining_ship": True, "capital_role": "capital_industrial"})


def test_pick_fleet_flagship_does_not_force_cyno():
    g = {"titan_slices": {"amarr": {"ship": {"111": 0.99, "10": 0.9, "11": 0.8, "12": 0.7, "101": 0.05}}}}
    fleet = pick_fleet(_C(), g, "amarr", n=4)
    assert "111" in fleet
    assert "101" not in fleet


def test_prepare_flagship_hangar_without_cyno_stays_out():
    g = {
        "titan_slices": {"amarr": {"ship": {"111": 0.9, "10": 0.5, "11": 0.4}, "equip": {}}},
        "stance": {"offense": 0.5, "formation": 0.1, "economy": 0.05, "logistics": 0.1, "speed_control": 0.1},
        "prepare": {"buy_n": 4, "fit_slots": 2, "hangar_keep_if_economy_ge": 0.9},
    }
    board = {"field_width": 8, "field_height": 4, "hangar_width": 8, "field_odd_row_extra": 0}
    rng = random.Random(1)
    pieces = prepare_pieces(_C(), g, "amarr", ["111", "10", "11"], board, rng)
    slots = {p["ship_id"]: p["slot"] for p in pieces}
    assert slots["111"] == "hangar"
    assert "101" not in slots
    fight = fleet_ids_from_pieces(pieces, _C.ships)
    assert "111" not in fight


def test_prepare_cyno_plus_flagship_jumps():
    g = {
        "titan_slices": {"amarr": {"ship": {"111": 0.9, "101": 0.5, "10": 0.4}, "equip": {}}},
        "stance": {"offense": 0.5, "formation": 0.1, "economy": 0.05, "logistics": 0.1, "speed_control": 0.1},
        "prepare": {"buy_n": 4, "fit_slots": 2, "hangar_keep_if_economy_ge": 0.9},
    }
    board = {"field_width": 8, "field_height": 4, "hangar_width": 8, "field_odd_row_extra": 0}
    rng = random.Random(1)
    pieces = prepare_pieces(_C(), g, "amarr", ["111", "101", "10"], board, rng)
    slots = {p["ship_id"]: p["slot"] for p in pieces}
    assert slots["111"] == "hangar"
    assert slots["101"] == "field"
    cyno = next(p for p in pieces if p["ship_id"] == "101")
    assert cyno["equips"] == []
    fight = fleet_ids_from_pieces(pieces, _C.ships)
    assert "111" in fight
    assert "101" in fight


def test_equip_goes_to_higher_ship_weight():
    class E(_C):
        equip_ids = ["gun_a", "gun_b"]
        equip_meta = {
            "gun_a": {"name": "A", "cost": 1, "line": "a"},
            "gun_b": {"name": "B", "cost": 1, "line": "b"},
        }

    g = {
        "titan_slices": {
            "amarr": {
                "ship": {"10": 0.9, "11": 0.2},
                "equip": {"gun_a": 0.95, "gun_b": 0.1},
            }
        },
        "stance": {"offense": 0.3, "formation": 0.2, "economy": 0.05, "logistics": 0.2, "speed_control": 0.1},
        "prepare": {"buy_n": 2, "fit_slots": 1, "hangar_keep_if_economy_ge": 0.99},
    }
    board = {"field_width": 8, "field_height": 4, "hangar_width": 8, "field_odd_row_extra": 0}
    pieces = prepare_pieces(E(), g, "amarr", ["10", "11"], board, random.Random(0))
    by = {p["ship_id"]: p["equips"] for p in pieces}
    assert any(e.startswith("gun_a:") for e in by["10"])
    assert not any(e.startswith("gun_a:") for e in by["11"])
