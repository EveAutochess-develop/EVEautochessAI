from eveac_ai.content import Content
from eveac_ai.ship import spawn_from_content
from eveac_ai.weapon_derive import attack_slot_count, derive_attack, get_module


def test_modules_table_loaded():
    c = Content()
    assert get_module(c, 456)
    assert float(c.modules["456"]["emDamage"]) > 0


def test_manned_dph_from_kit_not_cost():
    c = Content()
    hull = c.ships["10"]
    d = derive_attack(hull, c)
    slots = attack_slot_count(hull)
    kit = c.modules["456"]
    assert slots == 3
    assert d["damage"]["emp"] == round(float(kit["emDamage"]) * 3, 2)
    assert d["damage"]["thermal"] == round(float(kit["thermalDamage"]) * 3, 2)
    s1 = spawn_from_content(c, "10", 0, 1, star=1)
    s2 = spawn_from_content(c, "10", 0, 2, star=2)
    assert s1.damage["emp"] == d["damage"]["emp"]
    assert abs(s2.damage["emp"] - d["damage"]["emp"] * 2) < 0.02
    # old farm fake: cost*35 kinetic — must not appear
    assert s1.damage["kinetic"] == 0.0


def test_two_hulls_different_params():
    c = Content()
    a = spawn_from_content(c, "10", 0, 1)
    b = spawn_from_content(c, "50", 0, 2)
    assert a.max_shield != b.max_shield
    assert a.signature != b.signature
    assert a.scan != b.scan
    assert a.speed != b.speed
    assert a.damage != b.damage
    assert a.attack_duration != b.attack_duration or a.weapon_fx != b.weapon_fx


def test_unmanned_uses_baked_stars():
    c = Content()
    d = spawn_from_content(c, "1001", 0, 1, star=1)
    assert d.damage["emp"] == 36.0
    assert d.tracking > 0
