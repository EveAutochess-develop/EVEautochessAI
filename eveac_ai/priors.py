"""Generation 0: load LLM handbook priors, jitter 20 seats. No gaussian-from-scratch default."""

from __future__ import annotations

import copy
import json
import math
import random
from pathlib import Path
from typing import Any

from eveac_ai import SCHEMA_VER, STANCE_FLOOR, STANCE_IDS, TITAN_IDS
from eveac_ai.content import Content
from eveac_ai.prepare import is_shop_combat_hull

ROOT = Path(__file__).resolve().parents[1]


def scrub_genome_ships(genome: dict[str, Any], content: Content | None = None) -> dict[str, Any]:
    """Drop titans/wrecks/shop-ineligible keys from titan_slices.*.ship (AI_SELFPLAY genome gate)."""
    ships = (content.ships if content is not None else {}) or {}
    slices = genome.get("titan_slices") or {}
    for sl in slices.values():
        if not isinstance(sl, dict):
            continue
        raw = sl.get("ship") or {}
        if not isinstance(raw, dict):
            continue
        cleaned: dict[str, float] = {}
        for sid, w in raw.items():
            hull = ships.get(str(sid))
            if content is None:
                # No content: keep only if key looks like a normal shop id later; drop known wreck/titan ids.
                if str(sid) in ("201", "202", "203", "204", "205", "921", "922", "923", "924", "925"):
                    continue
                cleaned[str(sid)] = float(w)
                continue
            if is_shop_combat_hull(hull):
                cleaned[str(sid)] = float(w)
        sl["ship"] = cleaned
    return genome


def softmax(xs: list[float], floor: float = STANCE_FLOOR) -> list[float]:
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    raw = [e / s for e in exps]
    lifted = [max(floor, v) for v in raw]
    t = sum(lifted)
    return [v / t for v in lifted]


def load_bootstrap(path: Path | None = None, content: Content | None = None) -> dict[str, Any]:
    p = path or (ROOT / "priors" / "llm_bootstrap.genome.json")
    if not p.is_file():
        raise FileNotFoundError(f"missing LLM prior {p}; run python -m eveac_ai.bootstrap")
    genome = json.loads(p.read_text(encoding="utf-8"))
    return scrub_genome_ships(genome, content)


def derive_seat_genome(base: dict[str, Any], rng: random.Random, content: Content | None = None) -> dict[str, Any]:
    """Jitter a personality. Does not brand a titan onto the seat."""
    g = scrub_genome_ships(copy.deepcopy(base), content)
    g["origin"] = "llm_bootstrap"
    g.pop("active_titan", None)
    jitter = 0.04
    slices = g.get("titan_slices") or {}
    for tname, sl in slices.items():
        ships = sl.get("ship") or {}
        sl["ship"] = {k: round(max(0.05, min(0.99, float(v) + rng.uniform(-jitter, jitter))), 4) for k, v in ships.items()}
        equips = sl.get("equip") or {}
        sl["equip"] = {k: round(max(0.05, min(0.99, float(v) + rng.uniform(-jitter, jitter))), 4) for k, v in equips.items()}
    st = g.get("stance") or {}
    logits = [math.log(max(float(st.get(name, 0.2)), 1e-6)) + rng.uniform(-0.15, 0.15) for name in STANCE_IDS]
    mix = softmax(logits)
    g["stance"] = {STANCE_IDS[i]: round(mix[i], 4) for i in range(len(STANCE_IDS))}
    tp = g.get("titan_pick") or {t: 1.0 / len(TITAN_IDS) for t in TITAN_IDS}
    tlogits = [math.log(max(float(tp.get(t, 0.2)), 1e-6)) + rng.uniform(-0.2, 0.2) for t in TITAN_IDS]
    tmix = softmax(tlogits)
    pr = g.get("prepare") or {}
    g["prepare"] = {
        "buy_n": max(1, int(pr.get("buy_n", 4))),
        "fit_slots": max(1, min(3, int(pr.get("fit_slots", 2)))),
        "hangar_keep_if_economy_ge": round(
            max(0.05, min(0.95, float(pr.get("hangar_keep_if_economy_ge", 0.22)) + rng.uniform(-0.02, 0.02))),
            4,
        ),
    }
    return g


def pick_fleet(content: Content, genome: dict[str, Any], titan: str, n: int | None = None) -> list[str]:
    sl = (genome.get("titan_slices") or {}).get(titan) or {}
    weights: dict[str, float] = sl.get("ship") or {}
    take = int(n if n is not None else (genome.get("prepare") or {}).get("buy_n", 4))
    ranked: list[tuple[float, str]] = []
    for sid, hull in content.ships.items():
        if not is_shop_combat_hull(hull):
            continue
        w = float(weights.get(sid, 0.4))
        ranked.append((w, sid))
    ranked.sort(key=lambda kv: (-kv[0], kv[1]))
    ids = [sid for _, sid in ranked[: max(take, 1)]]
    return ids or ["10"]
