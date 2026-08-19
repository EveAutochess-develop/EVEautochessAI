"""Equip credit + genome equip write."""

from __future__ import annotations

from eveac_ai import telemetry_credit as tc


def test_equip_credit_splits_and_writes_genome():
    content = type(
        "C",
        (),
        {
            "ships": {"10": {"shop_eligible": True, "cost": 3}},
            "equip_meta": {"e1": {"cost": 2, "shop_pool": True}, "e2": {"cost": 2, "shop_pool": True}},
        },
    )()
    ship_credit = {"10": {"dmg": 1.0, "tank": 0.0, "repair": 0.0, "cap": 0.0, "w": 1.0, "c": 1.0, "kill": 0.0}}
    pieces = [{"slot": "field", "ship_id": "10", "equips": ["e1", "e2"], "star": 1}]
    old = tc.is_shop_combat_hull
    tc.is_shop_combat_hull = lambda h: True  # type: ignore
    try:
        eq = tc.equip_credit(content, pieces, ship_credit)
        assert "e1" in eq and "e2" in eq
        assert abs(eq["e1"]["w"] - 0.5) < 1e-6
        merged = tc.merge_ship_equip_credit(content, pieces, ship_credit)
        assert "10" in merged and "e1" in merged
        genome = {"titan_slices": {"amarr": {"ship": {"10": 0.4}, "equip": {"e1": 0.4, "e2": 0.4}}}}
        before = float(genome["titan_slices"]["amarr"]["equip"]["e1"])
        tc.apply_genome_delta(genome, "amarr", merged, True, content=content, source="natural")
        assert genome["titan_slices"]["amarr"]["equip"]["e1"] != before
    finally:
        tc.is_shop_combat_hull = old


def test_state_bank_rejects_lowsec_wrong_seats():
    from eveac_ai.state_bank import validate_league

    errs = validate_league(
        {"seats": [{"seat_id": 0, "board": {"level": 1, "pieces": [], "gold": 10}, "titan_hp": 300}]},
        mode="lowsec",
    )
    assert errs
