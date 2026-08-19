from eveac_ai.content import Content, load_config
from eveac_ai.pve import is_pvp_round, lock_creeps, pve_success, roll_pve_task, sleeper_pool


def test_nullsec_pvp_schedule():
    assert [is_pvp_round(r) for r in range(1, 9)] == [False, False, True, False, False, True, False, True]


def test_sleeper_pool_and_lock():
    c = Content(cfg=load_config())
    pool = sleeper_pool(c, 1)
    assert pool
    import random

    rng = random.Random(1)
    roster = lock_creeps(c, rng, gold=20, level=1, pop_limit=3, field_value=6)
    assert roster
    assert all(str(r["ship_id"]) in pool for r in roster)


def test_pve_success_eliminate_and_salvage():
    creeps_dead = {"ships": [{"ship_id": "221", "survived": False, "is_unmanned": False}]}
    creeps_live = {"ships": [{"ship_id": "221", "survived": True, "is_unmanned": False}]}
    player = {"ships": [{"ship_id": "211", "survived": True}]}
    assert pve_success(task="pve_eliminate", row_player=player, row_creep=creeps_dead, freighter_id="")
    assert not pve_success(task="pve_eliminate", row_player=player, row_creep=creeps_live, freighter_id="")
    assert pve_success(task="pve_salvage", row_player=player, row_creep=creeps_live, freighter_id="211")
    player_dead_f = {"ships": [{"ship_id": "211", "survived": False}]}
    assert not pve_success(task="pve_salvage", row_player=player_dead_f, row_creep=creeps_live, freighter_id="211")


def test_roll_pve_task_same_round_all_seats():
    a = roll_pve_task(None, 1, match_seed=42)
    b = roll_pve_task(None, 1, match_seed=42)
    c = roll_pve_task(None, 2, match_seed=42)
    assert a == b
    assert a in ("pve_eliminate", "pve_salvage")
    assert c in ("pve_eliminate", "pve_salvage")
