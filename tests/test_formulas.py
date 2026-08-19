from eveac_ai.formulas import turret_hit_chance, turret_hit_quality, lock_time_s, missile_damage_factor


def test_turret_at_optimal_stationary():
    p = turret_hit_chance(
        attacker_tracking=0.1,
        attacker_optimal_cells=10.0,
        attacker_falloff_cells=5.0,
        attacker_optimal_sig=40.0,
        target_speed=0.0,
        target_signature=40.0,
        distance_cells=10.0,
    )
    assert p == 0.99


def test_turret_range_falloff_hand():
    ## dist 20, opt 10, fo 5 cells → range_term = (40000-20000)/10000 = 2 → 0.5**4 = 0.0625
    p = turret_hit_chance(
        attacker_tracking=0.1,
        attacker_optimal_cells=10.0,
        attacker_falloff_cells=5.0,
        attacker_optimal_sig=40.0,
        target_speed=0.0,
        target_signature=40.0,
        distance_cells=20.0,
    )
    assert abs(p - 0.0625) < 1e-9


def test_hit_quality_miss_and_wrecking():
    assert turret_hit_quality(0.9, 0.5) == 0.0
    assert turret_hit_quality(0.005, 0.5) == 3.0
    assert abs(turret_hit_quality(0.2, 0.5) - 0.7) < 1e-9


def test_lock_time():
    assert abs(lock_time_s(400, 100, 40000.0) - 1.0) < 1e-9


def test_missile_factor_no_drf_is_sig_ratio():
    f = missile_damage_factor(80, 0, 40, 100, 0.0, 5.5)
    assert abs(f - 1.0) < 1e-9
