from eveac_ai.economy import interest_of, mining_gold_from_survivors, round_income_pre
from eveac_ai.content import Content, load_config


def test_interest_matches_client():
    econ = {"interest_divisor": 10, "interest_cap": 5, "interest_capped": True}
    assert interest_of(econ, 9) == 0
    assert interest_of(econ, 10) == 1
    assert interest_of(econ, 49) == 4
    assert interest_of(econ, 50) == 5
    assert interest_of(econ, 99) == 5


def test_mining_porpoise_not_self_buffed():
    c = Content(cfg=load_config())
    porp = {"ship_id": "136", "star": 1, "survived": True, "is_unmanned": False}
    ret = {"ship_id": "135", "star": 1, "survived": True, "is_unmanned": False}
    only_p = mining_gold_from_survivors(c, [porp])
    both = mining_gold_from_survivors(c, [porp, ret])
    assert only_p == 10
    assert both == 10 + int(25 * 1.2)


def test_round_income_includes_interest():
    c = Content(cfg=load_config())
    e = c.economy
    p = round_income_pre(e, gold_ref=40, round_i=3, won=True, win_streak=3, mining_g=0)
    assert p["interest"] == 4
    assert p["base"] == 5
    assert p["win"] == 1
    assert p["streak"] >= 1
