from types import SimpleNamespace

from eveac_ai.farm_lock import consume_stop_file
from eveac_ai.match20 import _json_safe, apply_seat_checkpoint, restore_leagues_from_checkpoint
from eveac_ai.nets.memory import hydrate_memory, match_global_obs
from eveac_ai.seat_prep import new_seat_board


def test_consume_stop_file(tmp_path):
    samples = tmp_path / "samples"
    samples.mkdir()
    assert consume_stop_file(samples) is False
    (samples / "farm.stop").write_text("1", encoding="utf-8")
    assert consume_stop_file(samples) is True
    assert not (samples / "farm.stop").is_file()
    assert consume_stop_file(samples) is False


def test_apply_seat_checkpoint_keeps_shop_keys():
    seat = {
        "seat_id": 0,
        "titan": "amarr",
        "board": new_seat_board(),
        "gold": 5,
        "alive": True,
        "titan_hp": 300,
        "wins": 0,
        "losses": 0,
        "fleet": [],
        "genome": {"x": 1},
        "memory": {},
        "frozen": False,
    }
    apply_seat_checkpoint(
        seat,
        {
            "seat_id": 0,
            "titan": "gallente",
            "titan_hp": 65,
            "wins": 8,
            "losses": 47,
            "alive": True,
            "gold": 14,
            "fleet": ["4", "14"],
            "board": {
                "gold": 14,
                "level": 8,
                "xp": 24,
                "win_streak": 0,
                "loss_streak": 2,
                "bag": ["nos_m"],
                "shop": ["4"],
                "token": 3,
                "pieces": [{"slot": "field", "ship_id": "4", "star": 1}],
            },
            "genome": {"x": 2},
            "frozen": True,
        },
    )
    assert seat["titan"] == "gallente"
    assert seat["titan_hp"] == 65
    assert seat["board"]["level"] == 8
    assert seat["board"]["shop_ships"] == ["4"]
    assert "shop_equips" in seat["board"]
    assert seat["frozen"] is True
    assert seat["genome"]["x"] == 2


def test_restore_leagues_skips_omitted_as_done():
    def _league(i: int):
        return SimpleNamespace(
            league_i=i,
            done=False,
            draft={},
            security_mode="nullsec",
            train_source="nullsec",
            force_pvp=False,
            match_id=f"gen59L{i}",
            seed=1,
            seats=[
                {
                    "seat_id": 0,
                    "titan": "amarr",
                    "board": new_seat_board(),
                    "gold": 5,
                    "alive": True,
                    "titan_hp": 300,
                    "wins": 0,
                    "losses": 0,
                    "fleet": [],
                    "genome": {},
                    "memory": {},
                }
            ],
        )

    leagues = [_league(0), _league(1)]
    blob = {
        "round_done": 54,
        "leagues": [
            {
                "league_i": 0,
                "done": False,
                "security_mode": "lowsec",
                "train_source": "lowsec",
                "seats": [
                    {
                        "seat_id": 0,
                        "titan": "caldari",
                        "titan_hp": 65,
                        "wins": 8,
                        "losses": 47,
                        "alive": True,
                        "gold": 14,
                        "fleet": ["4"],
                        "board": {"gold": 14, "level": 8, "xp": 24, "pieces": []},
                    }
                ],
            }
        ],
    }
    nxt = restore_leagues_from_checkpoint(leagues, blob)
    assert nxt == 55
    assert leagues[0].security_mode == "lowsec"
    assert leagues[0].force_pvp is True
    assert leagues[0].seats[0]["titan_hp"] == 65
    assert leagues[0].seats[0]["board"]["level"] == 8
    assert leagues[1].done is True


def test_hydrate_memory_from_str_set_and_json_list():
    m = hydrate_memory({"fought": "{0, 1}", "rounds": [[0.0]], "seen_field_cost": [3.0]})
    assert m["fought"] == {0, 1}
    m2 = hydrate_memory({"fought": [0, 2]})
    assert m2["fought"] == {0, 2}
    assert isinstance(_json_safe({"fought": {0, 1}})["fought"], list)
    seats = [{"alive": True, "titan": "amarr", "titan_hp": 300} for _ in range(2)]
    viewer = {"memory": {"fought": "{0}", "rounds": [], "seen_field_cost": [0.0] * 20}, "titan_hp": 300}
    vec = match_global_obs(seats=seats, viewer=viewer, rnd=1, n_seats=2, security_mode="lowsec")
    assert isinstance(vec, list)
    assert isinstance(viewer["memory"]["fought"], set)
