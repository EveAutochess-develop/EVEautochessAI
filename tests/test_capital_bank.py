"""Shop capital pool, cyno-key credit, capital-bank fuse."""

from __future__ import annotations

import random

from eveac_ai.capital_bank import fuse_league, fuse_seat
from eveac_ai.content import Content, load_config
from eveac_ai.prepare import is_cyno_flagship, is_covert_cyno
from eveac_ai.shop import in_capital_pool, roll_ship_shop
from eveac_ai.telemetry_credit import cyno_key_delta, eval_valence_delta, path_value, xp_shape_delta


def test_shop_capital_roll_zero_below_15():
    c = Content(cfg=load_config())
    rng = random.Random(0)
    hits = 0
    n = 400
    for _ in range(n):
        for sid in roll_ship_shop(c, level=14, titan="gallente", rng=rng):
            h = c.ships.get(str(sid)) or {}
            if is_cyno_flagship(h) or str(h.get("ship_group") or "") == "capital_industrial":
                hits += 1
    assert hits == 0


def test_shop_capital_roll_about_12pct_at_15():
    c = Content(cfg=load_config())
    rng = random.Random(1)
    cap = 0
    slots = 0
    for _ in range(800):
        shop = roll_ship_shop(c, level=15, titan="gallente", rng=rng)
        slots += len(shop)
        for sid in shop:
            if in_capital_pool(c.ships.get(str(sid)) or {}, c.economy):
                cap += 1
    rate = cap / max(1, slots)
    assert 0.04 < rate < 0.28


def test_cyno_key_positive_when_flag_hangar_and_cyno_field():
    content = type(
        "C",
        (),
        {
            "ships": {
                "101": {"capital_role": "covert_cyno"},
                "111": {"requires_cyno_entry": True, "capital_role": "dreadnought"},
            }
        },
    )()
    pieces = [
        {"slot": "hangar", "ship_id": "111"},
        {"slot": "field", "ship_id": "101"},
    ]
    assert cyno_key_delta(content, pieces) > 0
    lost = eval_valence_delta(content, {"ships": []}, pieces, False)
    assert lost >= 0 or lost > -0.12  # no waiting_for_godot when flag can jump


def test_waiting_for_godot_only_without_flag():
    content = type("C", (), {"ships": {"101": {"capital_role": "covert_cyno"}}})()
    pieces = [{"slot": "field", "ship_id": "101"}]
    assert eval_valence_delta(content, {"ships": []}, pieces, False) < 0


def test_xp_shape_and_path_value():
    assert xp_shape_delta(True, True) > 0
    assert xp_shape_delta(False, True) < 0
    high = path_value(level=12, field_n=12, lives=10, max_lives=15, med_level=7, med_pop=7)
    low = path_value(level=7, field_n=7, lives=3, max_lives=15, med_level=10, med_pop=10)
    assert high > low


def test_fuse_flagship_stays_hangar():
    c = Content(cfg=load_config())
    rng = random.Random(2)
    seat = {
        "titan": "amarr",
        "titan_hp": 300,
        "board": {
            "level": 4,
            "xp": 0,
            "gold": 8,
            "token": 2,
            "pieces": [{"token": 1, "ship_id": "10", "star": 1, "equips": [], "slot": "field", "x": 0, "z": 0}],
        },
    }
    fuse_seat(c, seat, rng, variant="flag_no_cyno", loss=20.0)
    assert 12 <= int(seat["board"]["level"]) <= 16
    assert 30 <= int(seat["board"]["gold"]) <= 100
    for p in seat["board"]["pieces"]:
        h = c.ships.get(str(p["ship_id"])) or {}
        if is_cyno_flagship(h):
            assert p["slot"] == "hangar"
    league = {"seats": [seat, dict(seat)]}
    fuse_league(c, league, rng)
    from eveac_ai.state_bank import validate_league

    assert not validate_league(league, mode="lowsec", content=c)


def test_fight_units_injects_hangar_flag_only_with_field_cyno():
    from eveac_ai.match20 import fight_units

    c = Content(cfg=load_config())
    ships = c.ships
    flag = next(sid for sid, h in ships.items() if is_cyno_flagship(h))
    cyno = next(sid for sid, h in ships.items() if is_covert_cyno(h))
    pieces = [
        {"ship_id": "10", "slot": "field", "x": 0, "z": 0, "star": 1, "equips": []},
        {"ship_id": cyno, "slot": "field", "x": 1, "z": 0, "star": 1, "equips": []},
        {"ship_id": flag, "slot": "hangar", "x": 0, "z": 0, "star": 1, "equips": []},
    ]
    units = fight_units(pieces, ships)
    ids = [u["ship_id"] for u in units]
    assert flag in ids
    assert all(u["cyno_hold"] for u in units if u["ship_id"] == flag)
    no_cyno = [p for p in pieces if p["ship_id"] != cyno]
    assert flag not in [u["ship_id"] for u in fight_units(no_cyno, ships)]


def test_farm_kernel_flagship_follows_cyno_delay():
    from eveac_ai.kernel import BattleKernel

    c = Content(cfg=load_config())
    k = BattleKernel(c)
    flag = next(sid for sid, h in c.ships.items() if is_cyno_flagship(h))
    pos = [{"ship_id": flag, "x": 0, "z": 0, "star": 1, "cyno_hold": True}]
    ships = k.spawn_fleet([flag], 0, 1, 0.0, pos)
    assert ships
    delay = float(c.match_flow.get("cyno_jump_delay_s") or 90.0)
    assert float(ships[0].hold_until) == delay

