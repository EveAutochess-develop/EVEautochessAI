"""Four nets: Ops may be per-seat; Shop/Fit/Place are one shared copy (batched forwards)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from eveac_ai.content import star_at, titan_max_hp, titan_pvp_loss, remaining_pvp_losses
from eveac_ai.economy import interest_of, sell_price, xp_demand
from eveac_ai.prepare import hull_size, is_cyno_flagship, is_covert_cyno, is_shop_combat_hull, legal_field_cells
from eveac_ai.ship import damage_sum
from eveac_ai.shop import roll_equip_shop, roll_ship_shop, synth_other_halves

TITANS = ["amarr", "caldari", "gallente", "minmatar", "angel"]
SIZES = ["S", "M", "L", "XL"]
MAX_STEPS = 24
MAX_CELLS = 64
N_SHIP_SHOP = 6
N_EQ_SHOP = 4
# shop actions: 6 ships + 4 equips + refresh + scanner + xp + sell + stop
N_SHOP_ACT = 6 + 4 + 5
CLIP = 0.2
ENTROPY = 0.01
LR = 3e-4


def default_collab() -> dict[str, Any]:
    return {
        "D": 32,
        "E": 32,
        "H": 256,
        "cap_ship": 256,
        "cap_equip": 256,
        "beta_shop": 1.0,
        "beta_fit": 0.7,
        "beta_place": 1.0,
        "alpha_ops": 0.30,
        "alpha_shop": 0.30,
        "alpha_fit": 0.15,
        "alpha_place": 0.25,
        "stop_grad_down": True,
        "layout_feedback": 0.2,
        "lam_econ": 1.2,
        "lam_hp": 0.40,  # PVP: d_hp is -1 if this round lost a life, else 0
        "lam_pop": 0.20,
        "lam_first_kill": 0.12,
        "lam_grade": 0.35,
        "lam_table": 0.25,
        "lam_dmg": 1.0,
        "lam_tank": 0.7,
        "lam_repair": 0.8,
        "lam_cap": 0.4,
        "lam_kill": 0.5,
        "lam_live": 0.3,
        "lam_xp_full_field": 0.12,
        "lam_cyno_key": 0.08,
        "lam_path": 0.55,
        "path_gamma": 0.85,
        "clip": CLIP,
        "entropy": ENTROPY,
    }


def _f(v: Any, d: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def _oh(i: int, n: int) -> list[float]:
    v = [0.0] * n
    if 0 <= i < n:
        v[i] = 1.0
    return v


def live_ship_vec(content: Any, ship_id: str, star: int = 1, residual: list[float] | None = None) -> list[float]:
    hull = content.ships.get(str(ship_id)) or {}
    st = star_at(hull, star)
    dmg = st.get("damage") if isinstance(st.get("damage"), dict) else {}
    dsum = damage_sum({k: _f(dmg.get(k)) for k in ("emp", "thermal", "kinetic", "explosive")})
    sz = hull_size(hull)
    race = str(hull.get("race") or "").lower()
    ema = residual if residual and len(residual) >= 4 else [0.0, 0.0, 0.0, 0.0]
    return [
        _f(hull.get("cost")) / 30.0,
        _f(st.get("shield_hp")) / 4000.0,
        _f(st.get("armor_hp")) / 4000.0,
        _f(st.get("structure_hp")) / 4000.0,
        _f(hull.get("speed")) / 400.0,
        _f(hull.get("signature_radius")) / 400.0,
        _f(hull.get("scan_resolution")) / 800.0,
        _f(hull.get("attack_cycle_s")) / 10.0,
        dsum / 400.0,
        1.0 if (hull.get("is_logistic") or st.get("is_logistic")) else 0.0,
        1.0 if is_covert_cyno(hull) else 0.0,
        1.0 if is_cyno_flagship(hull) else 0.0,
        *_oh(SIZES.index(sz) if sz in SIZES else 0, 4),
        *_oh(TITANS.index(race) if race in TITANS else 0, 5),
        float(star) / 3.0,
        min(1.0, _f(hull.get("mining_gold_per_round")) / 40.0) if hull.get("is_mining_ship") else min(1.0, len(hull.get("fetter_ids") or []) / 6.0),
        _f(hull.get("capacitor_capacity")) / 4000.0,
        float(ema[0]),
        float(ema[1]),
        float(ema[2]),
        float(ema[3]),
    ]


def live_equip_vec(content: Any, eid: str) -> list[float]:
    meta = content.equip_meta.get(str(eid).split(":", 1)[0]) or {}
    sz = str(meta.get("size") or "S").upper()
    ops = [str(e.get("op") or "") for e in (meta.get("effects") or []) if isinstance(e, dict)]
    return [
        _f(meta.get("cost")) / 20.0,
        *_oh(SIZES.index(sz) if sz in SIZES else 0, 4),
        1.0 if str(meta.get("shop_category")) == "repair" or any("repair" in o or "Repair" in o for o in ops) else 0.0,
        1.0 if any("mul" in o for o in ops) else 0.0,
        1.0 if meta.get("implant") else 0.0,
        float(len(ops)) / 6.0,
        1.0,
        1.0,
    ]


def _pad(v: list[float], n: int) -> list[float]:
    return (v + [0.0] * n)[:n]


def _mlp(nn, din: int, dout: int, h: int):
    return nn.Sequential(nn.Linear(din, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, dout))


def _size_ok(ship_sz: str, eq_sz: str) -> bool:
    order = {"S": 0, "M": 1, "L": 2, "XL": 3}
    return order.get(eq_sz, 0) <= order.get(ship_sz, 3)


class FourNetPack:
    """One file: titan + shared (ops/shop/fit/place) + match_global. Row-isolated Linear/SiLU."""

    SHIP_DIM = 28
    EQ_DIM = 12
    OPS_OBS = 64
    TITAN_IN = 21

    def __init__(self, *, device: object, dir_path: Path | None = None, collab: dict | None = None, **_kw: Any) -> None:
        import torch
        import torch.nn as nn

        from eveac_ai.nets.memory import MG_DIM

        self.torch = torch
        self.nn = nn
        self.device = device or torch.device("cpu")
        self.collab = dict(default_collab())
        if collab:
            self.collab.update(collab)
        self.dir = dir_path
        h = int(self.collab["H"])
        d = int(self.collab["D"])
        self.titan = _mlp(nn, self.TITAN_IN, 5, 64).to(self.device)
        self.match_global = _mlp(nn, MG_DIM, d, h).to(self.device)
        self.ops = _mlp(nn, self.OPS_OBS + d, d + 6 + 1, h).to(self.device)
        shop_in = N_SHIP_SHOP * self.SHIP_DIM + N_EQ_SHOP * self.EQ_DIM + d + 8
        self.shop = _mlp(nn, shop_in, N_SHOP_ACT, h).to(self.device)
        self.fit_scorer = _mlp(nn, self.EQ_DIM + self.SHIP_DIM + d + 8, 1, h).to(self.device)
        self.place = _mlp(nn, self.SHIP_DIM + 8 + MAX_CELLS + d, MAX_CELLS, h).to(self.device)
        params = (
            list(self.titan.parameters())
            + list(self.match_global.parameters())
            + list(self.ops.parameters())
            + list(self.shop.parameters())
            + list(self.fit_scorer.parameters())
            + list(self.place.parameters())
        )
        self.opt = torch.optim.Adam(params, lr=LR)
        self._shop_buf: list[tuple[Any, float]] = []
        self._fit_buf: list[tuple[Any, float]] = []
        self._place_buf: list[tuple[Any, float]] = []
        self._ops_buf: list[tuple[Any, Any, float]] = []
        self._titan_buf: list[tuple[Any, float]] = []
        self._mg_buf: list[tuple[Any, float]] = []
        self.last_backward: dict[str, Any] = {"n": 0, "skip": "init", "loss": 0.0}
        self.axis_ema: dict[str, list[float]] = {}
        if dir_path:
            self._try_load(dir_path)

    def _try_load(self, dpath: Path) -> None:
        torch = self.torch
        one = dpath / "behavior.nets.pt"
        if one.is_file():
            blob = torch.load(one, map_location=self.device)
            if isinstance(blob, dict):
                sh = blob.get("shared") or blob
                ti = blob.get("titan")
                if isinstance(sh, dict):
                    for key, mod in (("ops", self.ops), ("shop", self.shop), ("fit", self.fit_scorer), ("place", self.place)):
                        if key in sh:
                            try:
                                mod.load_state_dict(sh[key])
                            except Exception:
                                pass
                if blob.get("match_global"):
                    try:
                        self.match_global.load_state_dict(blob["match_global"])
                    except Exception:
                        pass
                if ti is not None:
                    try:
                        self.titan.load_state_dict(ti)
                    except Exception:
                        pass
                if blob.get("collab"):
                    self.collab.update(blob["collab"])
                ema = blob.get("axis_ema")
                if isinstance(ema, dict):
                    self.axis_ema = {str(k): list(v) for k, v in ema.items() if isinstance(v, list)}

    def save(self) -> None:
        if self.dir is None:
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        blob = {
            "schema_ver": "1",
            "titan": self.titan.state_dict(),
            "shared": {
                "ops": self.ops.state_dict(),
                "shop": self.shop.state_dict(),
                "fit": self.fit_scorer.state_dict(),
                "place": self.place.state_dict(),
            },
            "match_global": self.match_global.state_dict(),
            "collab": self.collab,
            "axis_ema": self.axis_ema,
        }
        self.torch.save(blob, self.dir / "behavior.nets.pt")
        (self.dir / "collab.json").write_text(json.dumps(self.collab, indent=2) + "\n", encoding="utf-8")

    def _t(self, xs: list[float]):
        return self.torch.tensor(xs, dtype=self.torch.float32, device=self.device)

    def ops_obs(
        self,
        content: Any,
        board: dict,
        genome: dict,
        titan: str,
        rnd: int,
        titan_hp: float,
        *,
        round_kind: str = "pvp",
        security_mode: str = "nullsec",
        seats: list | None = None,
    ) -> list[float]:
        econ = content.economy
        gold = int(board.get("gold") or 0)
        pieces = board.get("pieces") or []
        field = [p for p in pieces if p.get("slot") == "field"]
        hangar = [p for p in pieces if p.get("slot") == "hangar"]
        stance = genome.get("stance") or {}
        race = titan.replace("titan_", "")
        tp = getattr(content, "titan_pvp", None)
        lives = remaining_pvp_losses(titan_hp, tp)
        max_lives = max(1, remaining_pvp_losses(titan_max_hp(tp), tp))
        kind = str(round_kind or "pvp")
        mode = "lowsec" if str(security_mode) == "lowsec" else "nullsec"
        level = int(board.get("level") or 1)
        field_full = 1.0 if len(field) >= max(1, level) else 0.0
        need = max(1.0, float(xp_demand(econ, level)))
        med_lv = float(level)
        med_pop = float(len(field))
        if seats:
            alive = [s for s in seats if s.get("alive", True)]
            lvs = [int((s.get("board") or {}).get("level") or 1) for s in alive] or [level]
            pops = [
                sum(1 for p in ((s.get("board") or {}).get("pieces") or []) if p.get("slot") == "field")
                for s in alive
            ] or [len(field)]
            lvs.sort()
            pops.sort()
            med_lv = float(lvs[len(lvs) // 2])
            med_pop = float(pops[len(pops) // 2])
        last = board.get("_traj") if isinstance(board.get("_traj"), dict) else {}
        vec = [
            gold / 50.0,
            interest_of(econ, gold) / 5.0,
            level / 20.0,
            int(board.get("xp") or 0) / 40.0,
            rnd / 20.0,
            len(field) / 10.0,
            len(hangar) / 10.0,
            len(board.get("bag") or []) / 8.0,
            lives / float(max_lives),
            int(board.get("win_streak") or 0) / 8.0,
            int(board.get("loss_streak") or 0) / 8.0,
            sum(int(p.get("star") or 1) for p in field) / 20.0,
            sum(len(p.get("equips") or []) for p in pieces) / 12.0,
            1.0 if kind == "pvp" else 0.0,
            1.0 if kind == "pve_eliminate" else 0.0,
            1.0 if kind == "pve_salvage" else 0.0,
            1.0 if mode == "nullsec" else 0.0,
            1.0 if mode == "lowsec" else 0.0,
            *_oh(TITANS.index(race) if race in TITANS else 0, 5),
            float(stance.get("economy") or 0),
            float(stance.get("offense") or 0),
            float(stance.get("logistics") or 0),
            float(stance.get("speed_control") or 0),
            float(stance.get("formation") or 0),
            field_full,
            float(board.get("xp") or 0) / need,
            1.0 if gold >= int(econ.get("buy_exp_gold_cost") or 4) else 0.0,
            (level - med_lv) / 20.0,
            (len(field) - med_pop) / 10.0,
            float(last.get("d_level") or 0.0) / 5.0,
            float(last.get("d_pop") or 0.0) / 5.0,
            float(last.get("skipped_xp_when_full") or 0.0),
            1.0 if any(is_cyno_flagship(content.ships.get(str(p.get("ship_id"))) or {}) and p.get("slot") == "hangar" for p in pieces) else 0.0,
            1.0 if any(is_covert_cyno(content.ships.get(str(p.get("ship_id"))) or {}) and p.get("slot") == "field" for p in pieces) else 0.0,
        ]
        return _pad(vec, self.OPS_OBS)

    def titan_obs(self, genome: dict, *, census: dict | None, current: str | None, round_i: int) -> list[float]:
        from eveac_ai.titan_draft import titan_pick_vec

        pick = titan_pick_vec(genome)
        stance = genome.get("stance") or {}
        st = [
            float(stance.get("economy") or 0),
            float(stance.get("offense") or 0),
            float(stance.get("logistics") or 0),
            float(stance.get("speed_control") or 0),
            float(stance.get("formation") or 0),
        ]
        cen = [0.0] * 5
        if census:
            n = max(1, sum(int(census.get(t, 0)) for t in TITANS))
            cen = [int(census.get(t, 0)) / n for t in TITANS]
        cur = _oh(TITANS.index(current) if current in TITANS else -1, 5)
        return _pad(pick + st + cen + cur + [1.0 if round_i >= 2 else 0.0], self.TITAN_IN)

    def sample_titan(self, obs: list[float]):
        torch = self.torch
        logits = self.titan(self._t(obs).unsqueeze(0))[0]
        dist = torch.distributions.Categorical(logits=logits)
        idx = dist.sample()
        return int(idx.item()), dist.log_prob(idx)

    def encode_g(self, mg: list[float]):
        return self.match_global(self._t(mg).unsqueeze(0))[0]

    def _sample_masked(self, logits, mask: list[bool]):
        torch = self.torch
        m = self._t([0.0 if ok else -1e9 for ok in mask])
        dist = torch.distributions.Categorical(logits=logits + m)
        idx = dist.sample()
        return int(idx.item()), dist.log_prob(idx), dist.entropy()

    def prepare_seat(
        self,
        content: Any,
        genome: dict,
        titan: str,
        board: dict,
        rng,
        board_desc: dict,
        *,
        seat_id: int,
        rnd: int,
        titan_hp: float,
        seats: list | None = None,
        mg_vec: list[float] | None = None,
        round_kind: str = "pvp",
        security_mode: str = "nullsec",
    ) -> dict[str, Any]:
        from eveac_ai.nets.memory import match_global_obs
        from eveac_ai.seat_prep import try_merge

        torch = self.torch
        d = int(self.collab["D"])
        viewer = {"titan_hp": titan_hp, "memory": None, "titan": titan}
        if seats is not None and 0 <= seat_id < len(seats):
            viewer = seats[seat_id]
        if mg_vec is None:
            mg_vec = match_global_obs(
                seats=seats or [viewer],
                viewer=viewer,
                rnd=rnd,
                n_seats=len(seats or [viewer]),
                security_mode=security_mode,
            )
        g = self.encode_g(mg_vec)
        obs = self.ops_obs(
            content,
            board,
            genome,
            titan,
            rnd,
            titan_hp,
            round_kind=round_kind,
            security_mode=security_mode,
            seats=seats,
        )
        o = self.ops(torch.cat([self._t(obs), g]).unsqueeze(0))[0]
        intent = o[:d]
        gates = o[d : d + 6]
        adv_hat = o[d + 6]
        if self.collab.get("stop_grad_down", True):
            intent_d = intent.detach()
        else:
            intent_d = intent
        gprob = torch.sigmoid(gates)
        save_p = float(gprob[0].item())
        sell_p = float(gprob[1].item())
        scan_p = float(gprob[2].item())
        xp_p = float(gprob[3].item())
        _synth_p = float(gprob[4].item())
        refresh_p = float(gprob[5].item())
        econ = content.economy
        board["shop_ships"] = roll_ship_shop(content, level=board["level"], titan=titan, rng=rng)
        board["shop_equips"] = roll_equip_shop(content, level=board["level"], rng=rng)
        shop_lps = []
        shop_acts: list[dict[str, Any]] = []
        shop_ents = []
        steps = 0
        lv0 = int(board.get("level") or 1)
        pop0 = sum(1 for p in board["pieces"] if p.get("slot") == "field")
        bought_xp = False
        field_full_at_xp = False
        while steps < MAX_STEPS:
            steps += 1
            gold = int(board["gold"])
            feat = []
            ships = list(board["shop_ships"]) + [""] * N_SHIP_SHOP
            eqs = list(board["shop_equips"]) + [""] * N_EQ_SHOP
            for i in range(N_SHIP_SHOP):
                sid = ships[i] if i < len(ships) else ""
                feat.extend(_pad(live_ship_vec(content, sid, residual=self.axis_ema.get(str(sid))) if sid else [], self.SHIP_DIM))
            for i in range(N_EQ_SHOP):
                eid = eqs[i] if i < len(eqs) else ""
                feat.extend(_pad(live_equip_vec(content, eid) if eid else [], self.EQ_DIM))
            extra = [gold / 50.0, int(board["level"]) / 20.0, save_p, scan_p, xp_p, sell_p, float(len(board["pieces"])) / 12.0, 1.0]
            x = torch.cat([self._t(feat), intent_d, self._t(extra)])
            logits = self.shop(x.unsqueeze(0))[0]
            mask = [False] * N_SHOP_ACT
            for i in range(N_SHIP_SHOP):
                sid = ships[i] if i < len(ships) else ""
                hull = content.ships.get(str(sid)) or {}
                cost = int(float(hull.get("cost") or 99))
                mask[i] = bool(sid) and is_shop_combat_hull(hull) and cost <= gold
            for i in range(N_EQ_SHOP):
                eid = eqs[i] if i < len(eqs) else ""
                meta = content.equip_meta.get(str(eid)) or {}
                cost = int(float(meta.get("cost") or 99))
                mask[N_SHIP_SHOP + i] = bool(eid) and cost <= gold
            mask[10] = gold >= int(econ.get("refresh_cost") or 2)
            mask[11] = gold >= int(econ.get("ship_scanner_cost") or 50)
            mask[12] = gold >= int(econ.get("buy_exp_gold_cost") or 4) and int(board["level"]) < int(econ.get("player_level_cap") or 20)
            mask[13] = any(p.get("slot") == "hangar" and not is_cyno_flagship(content.ships.get(str(p["ship_id"]))) for p in board["pieces"])
            mask[14] = True
            bias = logits.clone()
            if refresh_p > 0.3:
                bias[10] = bias[10] + 1.2 * refresh_p
            if scan_p > 0.3:
                bias[11] = bias[11] + 2.0 * scan_p
            if xp_p > 0.3:
                bias[12] = bias[12] + 1.5 * xp_p
            if sell_p > 0.3:
                bias[13] = bias[13] + 1.5 * sell_p
            act, lp, ent = self._sample_masked(bias * float(self.collab["beta_shop"]), mask)
            shop_lps.append(lp)
            shop_ents.append(ent)
            if act == 14:
                shop_acts.append({"kind": "stop", "id": "", "lp": lp})
                break
            if act < 6:
                sid = ships[act]
                shop_acts.append({"kind": "buy_ship", "id": str(sid), "lp": lp})
                hull = content.ships.get(str(sid)) or {}
                board["gold"] -= int(float(hull.get("cost") or 0))
                tok = int(board["token"])
                board["token"] = tok + 1
                board["pieces"].append({"token": tok, "ship_id": str(sid), "star": 1, "equips": [], "slot": "hangar", "x": 0, "z": 0})
                board["shop_ships"][act] = ""
                try_merge(board["pieces"])
                continue
            if act < 10:
                ei = act - 6
                eid = eqs[ei]
                shop_acts.append({"kind": "buy_eq", "id": str(eid), "lp": lp})
                meta = content.equip_meta.get(str(eid)) or {}
                board["gold"] -= int(float(meta.get("cost") or 0))
                board["bag"].append(str(eid))
                if ei < len(board["shop_equips"]):
                    board["shop_equips"][ei] = ""
                continue
            if act == 10:
                shop_acts.append({"kind": "refresh", "id": "", "lp": lp})
                board["gold"] -= int(econ.get("refresh_cost") or 2)
                board["shop_ships"] = roll_ship_shop(content, level=board["level"], titan=titan, rng=rng)
                board["shop_equips"] = roll_equip_shop(content, level=board["level"], rng=rng)
                continue
            if act == 11:
                shop_acts.append({"kind": "scan", "id": "", "lp": lp})
                board["gold"] -= int(econ.get("ship_scanner_cost") or 50)
                halves = synth_other_halves(content, board["bag"] + [str(e).split(":")[0] for p in board["pieces"] for e in (p.get("equips") or [])])
                owned = [p["ship_id"] for p in board["pieces"]]
                board["shop_ships"] = roll_ship_shop(content, level=board["level"], titan=titan, rng=rng, owned_ids=owned, scanner=not halves)
                board["shop_equips"] = roll_equip_shop(content, level=board["level"], rng=rng, synth_halves=halves or None)
                continue
            if act == 12:
                field_n = sum(1 for p in board["pieces"] if p.get("slot") == "field")
                full = field_n >= max(1, int(board.get("level") or 1))
                shop_acts.append({"kind": "xp", "id": "", "lp": lp, "field_full": full})
                bought_xp = True
                field_full_at_xp = field_full_at_xp or full
                board["gold"] -= int(econ.get("buy_exp_gold_cost") or 4)
                board["xp"] += int(econ.get("buy_exp_amount") or 4)
                need = xp_demand(econ, board["level"])
                if board["xp"] >= need:
                    board["xp"] -= need
                    board["level"] += 1
                continue
            if act == 13:
                shop_acts.append({"kind": "sell", "id": "", "lp": lp})
                hang = [p for p in board["pieces"] if p.get("slot") == "hangar" and not is_cyno_flagship(content.ships.get(str(p["ship_id"])))]
                if hang:
                    p = hang[0]
                    cost = _f((content.ships.get(str(p["ship_id"])) or {}).get("cost"))
                    board["gold"] += sell_price(econ, cost)
                    board["pieces"].remove(p)
        fit_lps = self._fit(content, board, intent_d)
        place_lps = self._place(content, board, board_desc, intent_d)
        skipped_full = (not bought_xp) and pop0 >= max(1, lv0)
        board["_traj"] = {
            "d_level": int(board.get("level") or 1) - lv0,
            "d_pop": sum(1 for p in board["pieces"] if p.get("slot") == "field") - pop0,
            "skipped_xp_when_full": 1.0 if skipped_full else 0.0,
            "bought_xp": 1.0 if bought_xp else 0.0,
        }
        bern = torch.distributions.Bernoulli(logits=gates)
        gate_lp = bern.log_prob((gprob > 0.5).float()).sum()
        return {
            "obs": obs,
            "mg": mg_vec,
            "g": g,
            "adv_hat": adv_hat,
            "ops_lp": gate_lp,
            "shop_lps": shop_lps,
            "shop_acts": shop_acts,
            "fit_lps": fit_lps,
            "place_lps": place_lps,
            "field_cost": sum(_f((content.ships.get(str(p["ship_id"])) or {}).get("cost")) for p in board["pieces"] if p.get("slot") == "field"),
            "bought_xp": bought_xp,
            "field_full": pop0 >= max(1, lv0),
            "skipped_xp_when_full": skipped_full,
            "level": int(board.get("level") or 1),
            "field_n": sum(1 for p in board["pieces"] if p.get("slot") == "field"),
        }

    def _fit(self, content: Any, board: dict, intent) -> list:
        torch = self.torch
        lps = []
        beta = float(self.collab["beta_fit"])
        for eid in list(board.get("bag") or []):
            meta = content.equip_meta.get(str(eid).split(":", 1)[0]) or {}
            esz = str(meta.get("size") or "S").upper()
            ev = _pad(live_equip_vec(content, eid), self.EQ_DIM)
            cands = []
            for p in board["pieces"]:
                hull = content.ships.get(str(p["ship_id"])) or {}
                if is_covert_cyno(hull):
                    continue
                if len(p.get("equips") or []) >= 3:
                    continue
                if not _size_ok(hull_size(hull), esz):
                    continue
                cands.append(p)
            if not cands:
                continue
            scores = []
            for p in cands:
                sv = _pad(live_ship_vec(content, p["ship_id"], int(p.get("star") or 1), residual=self.axis_ema.get(str(p["ship_id"]))), self.SHIP_DIM)
                x = torch.cat([self._t(ev), self._t(sv), intent, self._t([len(p.get("equips") or []) / 3.0, 0, 0, 0, 0, 0, 0, 1.0])])
                scores.append(self.fit_scorer(x.unsqueeze(0))[0, 0] * beta)
            logits = torch.stack(scores)
            dist = torch.distributions.Categorical(logits=logits)
            idx = dist.sample()
            lps.append(dist.log_prob(idx))
            p = cands[int(idx.item())]
            p.setdefault("equips", []).append(f"{eid}:{meta.get('name') or eid}")
            board["bag"].remove(eid)
        return lps

    def _place(self, content: Any, board: dict, board_desc: dict, intent) -> list:
        from eveac_ai.seat_prep import _pop_cap

        torch = self.torch
        cells = legal_field_cells(board_desc)[:MAX_CELLS]
        n = len(cells)
        cap = _pop_cap(board)
        pieces = board["pieces"]
        for p in pieces:
            if p.get("slot") == "field" and is_cyno_flagship(content.ships.get(str(p["ship_id"]))):
                p["slot"] = "hangar"
        field = [p for p in pieces if p.get("slot") == "field"]
        hangar = [p for p in pieces if p.get("slot") == "hangar"]
        has_flag = any(is_cyno_flagship(content.ships.get(str(p["ship_id"]))) for p in hangar)
        cyno_field = any(is_covert_cyno(content.ships.get(str(p["ship_id"]))) for p in field)
        if has_flag and not cyno_field:
            hang_cyno = [p for p in hangar if is_covert_cyno(content.ships.get(str(p["ship_id"])))]
            if hang_cyno:
                if len(field) >= cap and field:
                    swap = next((p for p in reversed(field) if not is_covert_cyno(content.ships.get(str(p["ship_id"])))), None)
                    if swap is not None:
                        swap["slot"] = "hangar"
                        field.remove(swap)
                        hangar.append(swap)
                if len(field) < cap:
                    p = hang_cyno[0]
                    p["slot"] = "field"
                    hangar.remove(p)
                    field.append(p)
        while len(field) < cap:
            cand = [p for p in hangar if not is_cyno_flagship(content.ships.get(str(p["ship_id"])))]
            if has_flag:
                prefer = [p for p in cand if is_covert_cyno(content.ships.get(str(p["ship_id"])))]
                cand = prefer or cand
            if not cand:
                break
            p = cand[0]
            p["slot"] = "field"
            hangar.remove(p)
            field.append(p)
        while len(field) > cap:
            p = field[-1]
            p["slot"] = "hangar"
            field.remove(p)
        occ = [0.0] * MAX_CELLS
        taken: set[tuple[int, int]] = set()
        lps = []
        fps = [p for p in pieces if p.get("slot") == "field"]
        has_flag = any(is_cyno_flagship(content.ships.get(str(p["ship_id"]))) for p in pieces if p.get("slot") == "hangar")
        cyno_on = any(is_covert_cyno(content.ships.get(str(p["ship_id"]))) for p in fps)
        for p in fps:
            feat = _pad(live_ship_vec(content, p["ship_id"], int(p.get("star") or 1), residual=self.axis_ema.get(str(p["ship_id"]))), self.SHIP_DIM)
            extra = self._t(
                [
                    1.0 if has_flag else 0.0,
                    1.0 if is_covert_cyno(content.ships.get(str(p["ship_id"]))) else 0.0,
                    1.0 if cyno_on else 0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                ]
            )
            x = torch.cat([self._t(feat), extra, self._t(occ), intent])
            logits = self.place(x.unsqueeze(0))[0, :n]
            mask = [xz not in taken for xz in cells]
            act, lp, _ = self._sample_masked(logits * float(self.collab["beta_place"]), mask)
            lps.append(lp)
            p["x"], p["z"] = cells[act]
            taken.add(cells[act])
            occ[act] = 1.0
        return lps

    def _live_lp(self, lp):
        torch = self.torch
        if lp is None or not torch.is_tensor(lp):
            return None
        if lp.grad_fn is None and not bool(lp.requires_grad):
            return None
        return lp

    def remember(self, trace: dict[str, Any], adv: float, yhat: Any | None = None) -> None:
        from eveac_ai.telemetry_credit import buy_delta, xp_shape_delta

        clip = float(self.collab.get("clip") or CLIP)
        path = float(trace.get("path_value") or 0.0)
        mix = float(self.collab.get("lam_path") or 0.55)
        delta = (1.0 - mix) * float(adv) + mix * path
        delta = max(-clip * 5, min(clip * 5, delta))
        hat = trace.get("adv_hat")
        ops_lp = self._live_lp(trace.get("ops_lp"))
        if ops_lp is not None:
            if not self.torch.is_tensor(hat):
                hat = ops_lp * 0.0
            self._ops_buf.append((ops_lp, hat, delta))
        if trace.get("g") is not None and self._live_lp(trace.get("g")) is not None:
            self._mg_buf.append((trace["g"].sum() * 0.0, delta))
        credit = trace.get("credit") or {}
        acts = trace.get("shop_acts")
        sw = float(trace.get("source_weight") or 1.0)
        if acts:
            for act in acts:
                kind = str(act.get("kind") or "")
                kid = str(act.get("id") or "")
                if kind in ("buy_ship", "buy_eq"):
                    rec = credit.get(kid)
                    rest = 0.0 if rec is not None else delta
                elif kind == "xp":
                    rec = None
                    rest = xp_shape_delta(bool(act.get("field_full")), True, lam=float(self.collab.get("lam_xp_full_field") or 0.12))
                    rest = rest + 0.35 * path
                else:
                    rec = None
                    rest = delta
                d = buy_delta(self.collab, grade=delta, rec=rec, table_rest=rest)
                lp = self._live_lp(act.get("lp"))
                if lp is not None:
                    self._shop_buf.append((lp, d * sw))
        else:
            for lp in trace.get("shop_lps") or []:
                live = self._live_lp(lp)
                if live is not None:
                    self._shop_buf.append((live, delta))
        for lp in trace.get("fit_lps") or []:
            live = self._live_lp(lp)
            if live is not None:
                self._fit_buf.append((live, delta))
        place_bonus = float(trace.get("cyno_key") or 0.0)
        for lp in trace.get("place_lps") or []:
            live = self._live_lp(lp)
            if live is not None:
                self._place_buf.append((live, delta + place_bonus))
        for lp in trace.get("titan_lps") or []:
            live = self._live_lp(lp)
            if live is not None:
                self._titan_buf.append((live, delta))

    def backward_step(self) -> float:
        torch = self.torch
        a = self.collab
        loss = torch.zeros((), device=self.device)
        n = 0
        for lp, hat, delta in self._ops_buf:
            lp = self._live_lp(lp)
            if lp is None:
                continue
            loss = loss + float(a["alpha_ops"]) * (-lp * delta + 0.5 * (hat - delta) ** 2)
            n += 1
        for lp, delta in self._shop_buf:
            lp = self._live_lp(lp)
            if lp is None:
                continue
            loss = loss + float(a["alpha_shop"]) * (-lp * delta)
            n += 1
        for lp, delta in self._fit_buf:
            lp = self._live_lp(lp)
            if lp is None:
                continue
            loss = loss + float(a["alpha_fit"]) * (-lp * delta)
            n += 1
        for lp, delta in self._place_buf:
            lp = self._live_lp(lp)
            if lp is None:
                continue
            loss = loss + float(a["alpha_place"]) * (-lp * delta)
            n += 1
        for lp, delta in self._titan_buf:
            lp = self._live_lp(lp)
            if lp is None:
                continue
            loss = loss + 0.2 * (-lp * delta)
            n += 1
        self._ops_buf.clear()
        self._shop_buf.clear()
        self._fit_buf.clear()
        self._place_buf.clear()
        self._titan_buf.clear()
        self._mg_buf.clear()
        if n <= 0:
            self.last_backward = {"n": 0, "skip": "empty", "loss": 0.0}
            return 0.0
        loss = loss / n
        self.opt.zero_grad()
        try:
            loss.backward()
        except RuntimeError as exc:
            msg = str(exc).lower()
            if "second time" not in msg and "inplace" not in msg and "require grad" not in msg:
                raise
            print("nets skip backward: graph stale after weight update", flush=True)
            self.opt.zero_grad()
            self.last_backward = {"n": n, "skip": "stale", "loss": 0.0}
            return 0.0
        self.opt.step()
        out = float(loss.detach().item())
        self.last_backward = {"n": n, "skip": None, "loss": out}
        return out


def pair_advantage(collab: dict, *, won: bool, dmg_self: float, dmg_enemy: float, gold_self: float, gold_enemy: float, d_hp: float, pop_self: float, pop_enemy: float, draw: bool = False, first_kill: float = 900.0, wipe: bool = False, eval_delta: float = 0.0) -> float:
    from eveac_ai.telemetry_credit import table_grade

    return table_grade(
        collab,
        won=won,
        draw=draw,
        dmg_self=dmg_self,
        dmg_enemy=dmg_enemy,
        gold_self=gold_self,
        gold_enemy=gold_enemy,
        d_hp=d_hp,
        pop_self=pop_self,
        pop_enemy=pop_enemy,
        first_kill=first_kill,
        wipe=wipe,
        eval_delta=eval_delta,
    )
