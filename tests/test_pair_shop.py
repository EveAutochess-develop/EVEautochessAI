import random

from eveac_ai.match20 import fight_units, pair_alive
from eveac_ai.shop import roll_ship_shop
from eveac_ai.content import Content, load_config


def test_pair_alive_odd_bye():
    pairs, bye = pair_alive(list(range(5)), random.Random(1))
    assert bye is not None
    assert len(pairs) == 2
    ids = [bye] + [x for p in pairs for x in p]
    assert sorted(ids) == list(range(5))


def test_fight_units_keeps_duplicate_hulls():
    pieces = [
        {"token": 1, "ship_id": "10", "slot": "field", "x": 0, "z": 0, "star": 1, "equips": []},
        {"token": 2, "ship_id": "10", "slot": "field", "x": 1, "z": 0, "star": 2, "equips": ["e1"]},
    ]
    out = fight_units(pieces, {"10": {}})
    assert len(out) == 2
    assert out[1]["star"] == 2
    assert out[1]["equips"] == ["e1"]


def test_shop_rolls_six():
    c = Content(cfg=load_config())
    rng = random.Random(3)
    ships = roll_ship_shop(c, level=1, titan="amarr", rng=rng)
    assert len(ships) == 6
    assert all(str(s) in c.ships for s in ships)
