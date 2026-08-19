from eveac_ai.titan_draft import KEEP_BONUS, census, draft_two_rounds, second_round_probs, titan_pick_vec
from eveac_ai import TITAN_IDS
import random


def test_two_round_keep_is_legal_and_census_public():
    genomes = []
    for i in range(20):
        genomes.append({"titan_pick": {t: (0.6 if t == TITAN_IDS[i % 5] else 0.1) for t in TITAN_IDS}})
    rng = random.Random(7)
    d = draft_two_rounds(genomes, rng)
    assert len(d["round1"]) == 20
    assert sum(d["census"].values()) == 20
    assert d["kept"] >= 0
    # second round options always include current: keep bonus raises P(current) vs no-bonus clone
    counts = d["census"]
    g0 = genomes[0]
    cur = d["round1"][0]
    p_keep = second_round_probs(titan_pick_vec(g0), cur, counts, 20)
    p_nobonus = second_round_probs(titan_pick_vec(g0), "not-a-titan", counts, 20)
    idx = list(TITAN_IDS).index(cur)
    assert p_keep[idx] > p_nobonus[idx]
    assert KEEP_BONUS > 0


def test_census_only_counts():
    c = census(["amarr", "amarr", "angel"])
    assert c["amarr"] == 2
    assert c["angel"] == 1
    assert c["caldari"] == 0
