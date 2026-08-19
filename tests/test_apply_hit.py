from eveac_ai.ship import SimShip, apply_hit


def _hull(**kwargs) -> SimShip:
    s = SimShip(uid=1, ship_id="x", team=0)
    s.shield = 50
    s.armor = 50
    s.structure = 50
    s.max_shield = 50
    s.max_armor = 50
    s.max_structure = 50
    s.shield_resist = {"emp": 0, "thermal": 0, "kinetic": 0.5, "explosive": 0}
    s.armor_resist = {"emp": 0, "thermal": 0, "kinetic": 0, "explosive": 0}
    s.structure_resist = {"emp": 0, "thermal": 0, "kinetic": 0, "explosive": 0}
    for k, v in kwargs.items():
        setattr(s, k, v)
    return s


def test_layer_absorbed_no_pierce_needed():
    s = _hull()
    r = apply_hit(s, {"kinetic": 100}, pierce=True)
    assert abs(r["dealt"] - 50) < 1e-6
    assert s.shield == 0
    assert not r["destroyed"]


def test_pierce_into_armor():
    s = _hull()
    r = apply_hit(s, {"kinetic": 400}, pierce=True)
    ## shield: dealt=200 vs hp50 → absorb 50, keep=0.75, remaining kinetic=300
    ## armor resist 0: dealt=300 vs hp50 → absorb 50, keep leftover into structure
    assert s.shield == 0
    assert s.armor == 0
    assert r["dealt"] > 50
    assert s.destroyed or s.structure < 50
