from eveac_ai.nets.pack import FourNetPack, live_ship_vec
from eveac_ai.content import Content, load_config
from eveac_ai.nets.memory import MG_DIM, match_global_obs, new_memory


def test_one_shared_ops_not_modulelist():
    pack = FourNetPack(device=None)
    assert not hasattr(pack.ops, "__len__") or not isinstance(pack.ops, pack.nn.ModuleList)
    n_ops = sum(p.numel() for p in pack.ops.parameters())
    assert n_ops > 0
    assert sum(p.numel() for p in pack.shop.parameters()) > 0
    assert sum(p.numel() for p in pack.titan.parameters()) > 0
    assert sum(p.numel() for p in pack.match_global.parameters()) > 0


def test_titan_five_way():
    pack = FourNetPack(device=None)
    genome = {"titan_pick": {"amarr": 0.5, "caldari": 0.2, "gallente": 0.1, "minmatar": 0.1, "angel": 0.1}, "stance": {}}
    obs = pack.titan_obs(genome, census=None, current=None, round_i=1)
    assert len(obs) == pack.TITAN_IN
    logits = pack.titan(pack._t(obs).unsqueeze(0))[0]
    assert int(logits.numel()) == 5


def test_gold_changes_ops_obs():
    pack = FourNetPack(device=None)
    c = Content(cfg=load_config())
    g = {"stance": {}}
    b1 = {"gold": 5, "pieces": [], "bag": [], "level": 1, "xp": 0, "win_streak": 0, "loss_streak": 0}
    b2 = dict(b1)
    b2["gold"] = 40
    a = pack.ops_obs(c, b1, g, "amarr", 1, 100.0)
    b = pack.ops_obs(c, b2, g, "amarr", 1, 100.0)
    assert a != b


def test_row_shuffle_invariant():
    pack = FourNetPack(device=None)
    torch = pack.torch
    x = torch.randn(4, pack.OPS_OBS + int(pack.collab["D"]))
    y = pack.ops(x)
    perm = torch.tensor([2, 0, 3, 1])
    y2 = pack.ops(x[perm])
    assert torch.allclose(y[perm], y2, atol=1e-5)


def test_match_global_dim():
    seats = [{"alive": True, "titan": "amarr", "titan_hp": 100, "memory": new_memory()} for _ in range(20)]
    v = match_global_obs(seats=seats, viewer=seats[0], rnd=1, n_seats=20)
    assert len(v) == MG_DIM


def test_delayed_trace_backward_once():
    pack = FourNetPack(device=None)
    torch = pack.torch
    d = pack.OPS_OBS + int(pack.collab["D"])
    obs = torch.zeros(1, d)
    lp = pack.ops(obs)[0].sum()
    hat = lp * 0
    empty = {"ops_lp": lp, "adv_hat": hat, "shop_lps": [], "fit_lps": [], "place_lps": [], "titan_lps": []}
    pack.remember(empty, 0.2)
    loss1 = pack.backward_step()
    assert isinstance(loss1, float)
    lp2 = pack.ops(obs)[0].sum()
    empty2 = {"ops_lp": lp2, "adv_hat": lp2 * 0, "shop_lps": [], "fit_lps": [], "place_lps": [], "titan_lps": []}
    pack.remember(empty2, 0.1)
    loss2 = pack.backward_step()
    assert isinstance(loss2, float)


def test_second_backward_same_graph_skipped():
    pack = FourNetPack(device=None)
    torch = pack.torch
    obs = torch.zeros(1, pack.OPS_OBS + int(pack.collab["D"]))
    lp = pack.ops(obs)[0].sum()
    row = {"ops_lp": lp, "adv_hat": lp * 0, "shop_lps": [], "fit_lps": [], "place_lps": [], "titan_lps": []}
    pack.remember(row, 0.2)
    pack.backward_step()
    pack.remember(row, 0.1)
    assert pack.backward_step() == 0.0


def test_pair_forwards_one_step():
    pack = FourNetPack(device=None)
    torch = pack.torch
    obs = torch.zeros(1, pack.OPS_OBS + int(pack.collab["D"]))

    def row(lp):
        return {"ops_lp": lp, "adv_hat": lp * 0, "shop_lps": [], "fit_lps": [], "place_lps": [], "titan_lps": []}

    lp1 = pack.ops(obs)[0].sum()
    lp2 = pack.ops(obs)[0].sum()
    pack.remember(row(lp1), 0.1)
    pack.remember(row(lp2), 0.2)
    assert isinstance(pack.backward_step(), float)
    lp3 = pack.ops(obs)[0].sum()
    pack.remember(row(lp3), 0.15)
    assert isinstance(pack.backward_step(), float)


def test_remember_json_floats_no_crash():
    pack = FourNetPack(device=None)
    pack.remember(
        {"ops_lp": 0.0, "adv_hat": 0.0, "titan_lps": [0.1, 0.2], "shop_lps": [], "fit_lps": [], "place_lps": []},
        0.2,
    )
    assert pack.backward_step() == 0.0


def test_live_ship_not_tier():
    c = Content(cfg=load_config())
    sid = next(iter(c.ships))
    v = live_ship_vec(c, sid)
    assert len(v) == 28
    assert all(isinstance(x, float) for x in v)
