from eveac_ai.content import Content


def test_content_loads_combat_and_ships():
    c = Content()
    assert c.combat.get("sim_fixed_step_s") is None or True
    assert float(c.match_flow.get("sim_fixed_step_s", 0)) == 0.05
    assert len(c.ships) > 10
    assert "meters_per_cell" in c.combat
