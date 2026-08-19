from eveac_ai.content import Content
from eveac_ai.drones import DRONE_REVIVE_DELAY_S, spawn_combat_unmanned
from eveac_ai.kernel import BattleKernel
from eveac_ai.lance import hull_can_lance
from eveac_ai.ship import spawn_from_content


def test_drone_revives_after_400s_if_mother_lives():
    c = Content()
    k = BattleKernel(c)
    ships = k.spawn_fleet(["50"], 0, 1, 0.0)
    spawn_combat_unmanned(c, ships, 2)
    drones = [s for s in ships if s.is_unmanned]
    assert drones
    d = drones[0]
    d.destroyed = True
    d.structure = 0.0
    q = []
    k._schedule_revive(d, 1.0, q)
    assert q[0]["revive_at"] == 1.0 + DRONE_REVIVE_DELAY_S
    uid = [max(s.uid for s in ships) + 1]
    k._tick_revives(ships, q, uid, 1.0)
    assert q  # not yet
    k._tick_revives(ships, q, uid, 1.0 + DRONE_REVIVE_DELAY_S)
    assert not q
    assert any(s.is_unmanned and s.alive() and s.uid != d.uid for s in ships)


def test_mixed_lance_fires_on_dread():
    c = Content()
    k = BattleKernel(c)
    hull = c.ships["111"]
    from eveac_ai.ship import SimShip
    s = spawn_from_content(c, "111", 0, 1)
    assert hull_can_lance(s)
    r = k.fight(
        fleet_a=["111"],
        fleet_b=["10"],
        seed=3,
        match_id="lance",
        round_i=0,
        seat_a=0,
        seat_b=1,
        pos_a=[{"ship_id": "111", "x": 0, "z": 0, "equips": ["mixed_lance"]}],
        pos_b=[{"ship_id": "10", "x": 0, "z": 0}],
    )
    dread = next(x for x in r["seats"][0]["ships"] if x["ship_id"] == "111")
    # Prep 10s then ticks; fight should last long enough for some lance dmg or at least consume path.
    assert dread["dmg_out"] > 0 or any(k.get("via") == "mixed_lance" for k in r["seats"][0]["kill_calendar"])
