from eveac_ai.ranking import (
    TIER_ORDER,
    format_ranking_delta,
    is_titan_hull,
    item_key,
    quantize_ships,
    rankable_ship_ids,
    ranking_index,
    spend_catalog,
)


def test_ranking_excludes_titans_and_has_tu():
    ships = {
        "4": {"name": "富豪级", "cost": 2},
        "6": {"name": "龙骑兵级", "cost": 3},
        "201": {"name": "神使级", "capital_role": "titan", "ship_group": "titan", "cost": 99},
    }
    ids = rankable_ship_ids(ships)
    assert "201" not in ids
    assert "4" in ids
    g = {"content_rev": "t", "titan_ids": ["amarr"], "titan_slices": {"amarr": {"ship": {"4": 0.9, "6": 0.2}}}}
    table = quantize_ships(g, list(ships.keys()), seen_ids={"4"}, hulls=ships)
    listed = []
    for k in TIER_ORDER:
        listed.extend(table["tiers"][k]["ships"])
    assert "201" not in listed
    assert set(listed) == {"4", "6"}
    assert "6" in table["tiers"]["TU"]["ships"]
    assert "T0" in table["tier_order"] and "T48" in table["tier_order"] and table["tier_order"][-1] == "TU"


def test_ships_and_equips_share_one_ladder():
    ships = {"4": {"name": "富豪级", "cost": 2}}
    equips = {"gun_a": {"name": "小炮", "cost": 1.0}, "free": {"name": "零费", "cost": 0.0}}
    cat = spend_catalog(ships, equips)
    assert ("ship", "4") in cat
    assert ("equip", "gun_a") in cat
    assert ("equip", "free") not in cat
    g = {
        "titan_ids": ["amarr"],
        "titan_slices": {"amarr": {"ship": {"4": 0.2}, "equip": {"gun_a": 0.95}}},
    }
    table = quantize_ships(
        g,
        ["4"],
        seen_ids={item_key("ship", "4"), item_key("equip", "gun_a")},
        hulls=ships,
        equip_meta=equips,
    )
    t0 = table["tiers"]["T0"]["items"]
    assert t0[0]["kind"] == "equip" and t0[0]["id"] == "gun_a"
    mixed = []
    for k in TIER_ORDER:
        mixed.extend((it["kind"], it["id"]) for it in table["tiers"][k]["items"])
    assert mixed.count(("ship", "4")) == 1
    assert mixed.count(("equip", "gun_a")) == 1


def test_same_weight_unlimited_and_tu_unranked():
    """Same weight → same tier with no item cap; unused → TU only."""
    ships = {str(i): {"name": f"s{i}", "cost": 1} for i in range(12)}
    w = {str(i): 0.8 for i in range(10)}
    w["10"] = 0.1
    g = {"titan_ids": ["amarr"], "titan_slices": {"amarr": {"ship": w}}}
    table = quantize_ships(g, list(ships.keys()), seen_ids={str(i) for i in range(11)}, hulls=ships)
    t0 = table["tiers"]["T0"]["ships"]
    assert len(t0) == 10
    assert set(t0) == {str(i) for i in range(10)}
    assert "10" in table["tiers"]["T1"]["ships"]
    assert "11" in table["tiers"]["TU"]["ships"]
    idx = ranking_index(table)
    assert idx[item_key("ship", "0")] == "T0"
    assert idx[item_key("ship", "9")] == "T0"
    assert idx[item_key("ship", "11")] == "TU"
    old = {item_key("ship", "0"): "T0", item_key("ship", "11"): "TU"}
    new = {item_key("ship", "0"): "T0", item_key("ship", "11"): "T3"}
    delta = format_ranking_delta(old, new, {"11": "s11"}, gen=1)
    assert "TU → T3" in delta
    assert "#" not in delta


def test_is_titan_hull():
    assert is_titan_hull({"capital_role": "titan"})
    assert not is_titan_hull({"capital_role": "carrier"})
