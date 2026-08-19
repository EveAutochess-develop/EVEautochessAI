import json
from pathlib import Path

from eveac_ai.content import Content
from eveac_ai.kernel import BattleKernel


def test_1v1_deterministic():
    fixture = json.loads((Path(__file__).parent / "golden" / "1v1.json").read_text(encoding="utf-8"))
    k = BattleKernel(Content())
    a = k.fight(
        fleet_a=fixture["fleet_a"],
        fleet_b=fixture["fleet_b"],
        seed=int(fixture["seed"]),
        match_id="golden-1v1",
        round_i=0,
        seat_a=0,
        seat_b=1,
    )
    b = k.fight(
        fleet_a=fixture["fleet_a"],
        fleet_b=fixture["fleet_b"],
        seed=int(fixture["seed"]),
        match_id="golden-1v1",
        round_i=0,
        seat_a=0,
        seat_b=1,
    )
    assert a["seats"][0]["won"] == b["seats"][0]["won"]
    assert a["seats"][1]["won"] == b["seats"][1]["won"]
    ka = [x["victim"] for x in a["seats"][0]["kill_calendar"] + a["seats"][1]["kill_calendar"]]
    kb = [x["victim"] for x in b["seats"][0]["kill_calendar"] + b["seats"][1]["kill_calendar"]]
    assert ka == kb
    assert a["schema_ver"] == "1"
    assert a["seats"][0]["ships"]
    assert any(s["dmg_out"] > 0 or s["dmg_in"] > 0 for s in a["seats"][0]["ships"] + a["seats"][1]["ships"])
