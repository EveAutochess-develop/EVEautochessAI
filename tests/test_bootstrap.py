from eveac_ai.bootstrap import build_genome
from eveac_ai.content import Content


def test_bootstrap_has_five_slices_and_stances():
    g = build_genome(Content())
    assert g["origin"] == "llm_bootstrap"
    for t in ("amarr", "caldari", "gallente", "minmatar", "angel"):
        assert t in g["titan_slices"]
        assert g["titan_slices"][t]["ship"]
    for s in ("economy", "offense", "logistics", "speed_control", "formation"):
        assert s in g["stance"]
        assert g["stance"][s] >= 0.05
    assert "titan_pick" in g
    assert abs(sum(g["titan_pick"].values()) - 1.0) < 0.02
    assert "active_titan" not in g
