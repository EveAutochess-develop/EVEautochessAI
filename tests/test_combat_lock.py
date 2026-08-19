from eveac_ai.content import Content
from eveac_ai.drones import drone_spawn_policy, spawn_combat_unmanned
from eveac_ai.kernel import BattleKernel
from eveac_ai.rng import FarmRng
from eveac_ai.ship import spawn_from_content


def test_content_loads_unmanned_templates():
    c = Content()
    assert "1001" in c.ships
    assert c.ships["1001"].get("is_unmanned") is True
    assert c.ships["1001"].get("unmanned_kind") == "combat_drone"


def test_battleship_spawns_two_heavies():
    c = Content()
    hull = c.ships["50"]
    pol = drone_spawn_policy(
        hull,
        race=str(hull.get("race", "caldari")).lower(),
        group="battleship",
        ship_id=50,
        is_logistic=False,
    )
    assert pol["count"] == 2
    assert pol["drone_id"] == 1012
    mother = spawn_from_content(c, "50", 0, 1, x=0.0, z=0.0)
    ships = [mother]
    spawn_combat_unmanned(c, ships, 2)
    drones = [s for s in ships if s.is_unmanned]
    assert len(drones) == 2
    assert all(s.ship_id == "1012" for s in drones)
    assert all(s.mother_uid == 1 for s in drones)


def test_excavator_does_not_lock():
    c = Content()
    k = BattleKernel(c)
    r = k.fight(fleet_a=["138"], fleet_b=["10"], seed=1, match_id="ex", round_i=0, seat_a=0, seat_b=1)
    excav = [s for s in r["seats"][0]["ships"] if s.get("unmanned_kind") == "mining_excavator"]
    assert excav
    assert all(s.get("dmg_out", 0) == 0 for s in excav)


def test_lock_holds_until_periodic():
    c = Content()
    k = BattleKernel(c)
    ships = k.spawn_fleet(["50"], 0, 1, 0.0) + k.spawn_fleet(["61"], 1, 100, 12.0)
    spawn_combat_unmanned(c, ships, max(s.uid for s in ships) + 1)
    rng = FarmRng(7)
    lock_tl: list = []

    def living(team=None):
        return [s for s in ships if s.alive() and (team is None or s.team == team)]

    s = ships[0]
    t = 0.0
    while t < 1.0:
        t += k.dt
        lookup = {x.uid: x for x in ships}
        k._update_targeting(s, living, lookup, False, 0.5, t, rng, lock_tl)
    first = s.lock_uid
    assert first != 0
    for _ in range(40):
        t += k.dt
        lookup = {x.uid: x for x in ships}
        k._update_targeting(s, living, lookup, False, 0.5, t, rng, lock_tl)
    assert s.lock_uid == first
