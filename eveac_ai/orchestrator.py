"""One generation of 20 nullsec AI seats. Keep top 3. CPU kernel by default."""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eveac_ai.content import Content, load_config
from eveac_ai.kernel import BattleKernel
from eveac_ai.device import resolve_device
from eveac_ai.prepare import is_shop_combat_hull
from eveac_ai.priors import derive_seat_genome, load_bootstrap, pick_fleet, scrub_genome_ships
from eveac_ai.titan_draft import draft_two_rounds, draft_two_rounds_torch

SCHEMA_VER = "1"


def softmax(xs: list[float], floor: float = 0.05) -> list[float]:
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    raw = [e / s for e in exps]
    lifted = [max(floor, v) for v in raw]
    t = sum(lifted)
    return [v / t for v in lifted]


def pair_seats(n: int) -> list[tuple[int, int]]:
    return [(i, i + 1) for i in range(0, n, 2)]


from eveac_ai.ranking import N_STRENGTH, quantize_ships


def quantize_table(genome: dict, ships: list[str], equips: list[str], n_tiers: int, seen_ids: set[str] | None = None, ship_names: dict[str, str] | None = None) -> dict:
    _ = equips
    n = int(n_tiers) if n_tiers else N_STRENGTH
    if n != N_STRENGTH:
        n = N_STRENGTH
    return quantize_ships(genome, ships, seen_ids=seen_ids, content_rev=str(genome.get("content_rev", "")), ship_names=ship_names)


def blend_genomes(
    elites: list[dict],
    titans: list[str],
    stances: list[str],
    content: Content | None = None,
) -> dict:
    import json as _json

    base = _json.loads(_json.dumps(elites[0]))
    n = float(len(elites))
    for titan in titans:
        ships: set[str] = set()
        equips: set[str] = set()
        for g in elites:
            sl = (g.get("titan_slices") or {}).get(titan) or {}
            ships.update((sl.get("ship") or {}).keys())
            equips.update((sl.get("equip") or {}).keys())
        ship_out: dict[str, float] = {}
        equip_out: dict[str, float] = {}
        for sid in sorted(ships, key=str):
            if content is not None and not is_shop_combat_hull(content.ships.get(str(sid))):
                continue
            acc = 0.0
            for g in elites:
                sl = (g.get("titan_slices") or {}).get(titan) or {}
                acc += float((sl.get("ship") or {}).get(sid, 0.0))
            ship_out[sid] = round(acc / n, 4)
        for eid in sorted(equips):
            acc = 0.0
            for g in elites:
                sl = (g.get("titan_slices") or {}).get(titan) or {}
                acc += float((sl.get("equip") or {}).get(eid, 0.0))
            equip_out[eid] = round(acc / n, 4)
        base["titan_slices"][titan] = {"ship": ship_out, "equip": equip_out}
    stance_acc = {s: 0.0 for s in stances}
    for g in elites:
        st = g.get("stance") or {}
        for s in stances:
            stance_acc[s] += float(st.get(s, 0.0))
    mix = softmax([stance_acc[s] / n for s in stances])
    base["stance"] = {s: round(mix[i], 4) for i, s in enumerate(stances)}
    tp_acc = {t: 0.0 for t in titans}
    for g in elites:
        tp = g.get("titan_pick") or {}
        for t in titans:
            tp_acc[t] += float(tp.get(t, 1.0 / max(len(titans), 1)))
    tmix = softmax([tp_acc[t] / n for t in titans])
    base["titan_pick"] = {t: round(tmix[i], 4) for i, t in enumerate(titans)}
    knobs = {"buy_n": 0.0, "fit_slots": 0.0, "hangar_keep_if_economy_ge": 0.0}
    for g in elites:
        pr = g.get("prepare") or {}
        knobs["buy_n"] += float(pr.get("buy_n", 4))
        knobs["fit_slots"] += float(pr.get("fit_slots", 2))
        knobs["hangar_keep_if_economy_ge"] += float(pr.get("hangar_keep_if_economy_ge", 0.22))
    base["prepare"] = {
        "buy_n": max(1, int(round(knobs["buy_n"] / n))),
        "fit_slots": max(1, min(3, int(round(knobs["fit_slots"] / n)))),
        "hangar_keep_if_economy_ge": round(knobs["hangar_keep_if_economy_ge"] / n, 4),
    }
    base.pop("active_titan", None)
    base["origin"] = "farm"
    return scrub_genome_ships(base, content)


def run_fight(backend: str, kernel: BattleKernel | None, fleet_a: list[str], fleet_b: list[str], seat_a: dict, seat_b: dict, match_id: str) -> dict:
    if backend in ("cpu_stub", "gpu_stub"):
        import battle_stub

        return battle_stub.fight(backend=backend, match_id=match_id, round_i=0, seat_a=seat_a, seat_b=seat_b)
    if kernel is None:
        raise RuntimeError("cpu backend requires BattleKernel")
    seed = 20260815 + int(seat_a["seat_id"]) * 17 + int(seat_b["seat_id"])
    return kernel.fight(
        fleet_a=fleet_a,
        fleet_b=fleet_b,
        seed=seed,
        match_id=match_id,
        round_i=0,
        seat_a=int(seat_a["seat_id"]),
        seat_b=int(seat_b["seat_id"]),
    )


def main() -> None:
    cfg = load_config()
    content = Content(cfg=cfg)
    backend = str(cfg.get("battle_backend", "cpu"))
    n_seats = int(cfg.get("seats", 20))
    keep = int(cfg.get("keep_top", 3))
    titans = list(cfg.get("titan_ids") or ["amarr", "caldari", "gallente", "minmatar", "angel"])
    stances = list(cfg.get("stance_ids") or ["economy", "offense", "logistics", "speed_control", "formation"])
    n_tiers = int(cfg.get("tier_count", 8))
    out_dir = ROOT / cfg.get("out_dir", "samples")
    out_dir.mkdir(parents=True, exist_ok=True)

    prior = load_bootstrap(content=content)
    rng = random.Random(20260815)
    infer = resolve_device("gpu" if backend == "gpu" else ("cpu" if backend == "cpu" else "auto"))
    kernel = BattleKernel(content) if backend in ("cpu", "gpu") else None
    gpu_k = None
    if backend == "gpu":
        from eveac_ai.gpu_kernel import GpuBattleKernel

        gpu_k = GpuBattleKernel(content, infer.torch_device)

    seats = []
    genomes = [derive_seat_genome(prior, rng, content) for _ in range(n_seats)]
    if infer.kind == "cuda" and infer.torch_device is not None:
        draft = draft_two_rounds_torch(genomes, 20260815, infer.torch_device)
    else:
        draft = draft_two_rounds(genomes, rng)
    for i in range(n_seats):
        titan = draft["round2"][i]
        genome = genomes[i]
        fleet = pick_fleet(content, genome, titan, n=4)
        seats.append(
            {
                "seat_id": i,
                "titan": titan,
                "titan_round1": draft["round1"][i],
                "genome": genome,
                "fleet": fleet,
                "titan_hp": 100.0,
                "wins": 0,
                "hint": 0.0,
                "memory": {
                    "schema_ver": SCHEMA_VER,
                    "seat_id": i,
                    "last_engagement": None,
                    "scout_intel": [],
                    "opponent_profiles": {},
                },
            }
        )

    match_id = "gpu-gen-0" if backend == "gpu" else "cpu-gen-0"
    last_pack = None
    pairs = pair_seats(n_seats)
    if gpu_k is not None:
        jobs = [
            {
                "fleet_a": seats[a]["fleet"],
                "fleet_b": seats[b]["fleet"],
                "seed": 20260815 + a * 17 + b,
                "match_id": match_id,
                "round_i": 0,
                "seat_a": a,
                "seat_b": b,
            }
            for a, b in pairs
        ]
        packs = gpu_k.fight_batch(jobs)
        for pack in packs:
            last_pack = pack
            for row in pack["seats"]:
                sid = int(row["seat_id"])
                seats[sid]["titan_hp"] = float(row["titan_hp"])
                seats[sid]["hint"] = float(row.get("rank_hint") or 0.0)
                if row["won"]:
                    seats[sid]["wins"] += 1
                seats[sid]["memory"]["last_engagement"] = {"round": 0, "backend": pack.get("backend")}
    else:
        for a, b in pairs:
            pack = run_fight(backend, kernel, seats[a]["fleet"], seats[b]["fleet"], seats[a], seats[b], match_id)
            last_pack = pack
            for row in pack["seats"]:
                sid = int(row["seat_id"])
                seats[sid]["titan_hp"] = float(row["titan_hp"])
                seats[sid]["hint"] = float(row.get("rank_hint") or 0.0)
                if row["won"]:
                    seats[sid]["wins"] += 1
                seats[sid]["memory"]["last_engagement"] = {"round": 0, "backend": pack.get("backend")}

    ranked = sorted(seats, key=lambda s: (-s["wins"], -s["titan_hp"], -s["hint"], s["seat_id"]))
    elites = ranked[:keep]
    export = blend_genomes([e["genome"] for e in elites], titans, stances, content=content)
    export["content_rev"] = content.rev
    table = quantize_table(export, list(content.ships.keys()), content.equip_ids, n_tiers)

    (out_dir / "behavior.genome.json").write_text(json.dumps(export, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "weights_table.json").write_text(json.dumps(table, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tel_path = out_dir / "telemetry_sample.json"
    if last_pack:
        tel_path.write_text(json.dumps(last_pack, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = {
        "schema_ver": SCHEMA_VER,
        "backend": backend,
        "content_rev": content.rev,
        "data_dir": str(content.dir),
        "prior": "priors/llm_bootstrap.genome.json",
        "seats": n_seats,
        "keep_top": keep,
        "elite_seat_ids": [e["seat_id"] for e in elites],
        "elite_wins": [e["wins"] for e in elites],
        "titan_draft": {"census": draft["census"], "round1": draft["round1"], "round2": draft["round2"], "kept": draft["kept"]},
        "infer_device": infer.name,
        "fleets": {str(s["seat_id"]): s["fleet"] for s in seats},
        "wrote": ["behavior.genome.json", "weights_table.json", "telemetry_sample.json"],
    }
    (out_dir / "generation_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"generation ok backend={backend} elites={[e['seat_id'] for e in elites]}")
    print(f"wrote {out_dir / 'behavior.genome.json'}")


if __name__ == "__main__":
    main()
