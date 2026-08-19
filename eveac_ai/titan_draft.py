"""Two-round titan pick: blind → census → keep-or-switch. No seat branding."""

from __future__ import annotations

import math
import random
from typing import Any

from eveac_ai import STANCE_FLOOR, TITAN_IDS


KEEP_BONUS = 0.45
CROWD_K = 1.6


def titan_pick_vec(genome: dict[str, Any]) -> list[float]:
    raw = genome.get("titan_pick") or {}
    xs = [max(1e-6, float(raw.get(t, 1.0 / len(TITAN_IDS)))) for t in TITAN_IDS]
    s = sum(xs)
    return [v / s for v in xs]


def census(picks: list[str]) -> dict[str, int]:
    out = {t: 0 for t in TITAN_IDS}
    for p in picks:
        if p in out:
            out[p] += 1
    return out


def _sample(probs: list[float], rng: random.Random) -> str:
    x = rng.random()
    acc = 0.0
    for i, p in enumerate(probs):
        acc += p
        if x <= acc:
            return TITAN_IDS[i]
    return TITAN_IDS[-1]


def softmax_logits(logits: list[float], floor: float = STANCE_FLOOR) -> list[float]:
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s = sum(exps)
    raw = [e / s for e in exps]
    lifted = [max(floor, v) for v in raw]
    t = sum(lifted)
    return [v / t for v in lifted]


def second_round_probs(pick: list[float], current: str, counts: dict[str, int], n_seats: int) -> list[float]:
    expected = n_seats / float(len(TITAN_IDS))
    logits: list[float] = []
    for i, t in enumerate(TITAN_IDS):
        crowd = (float(counts.get(t, 0)) - expected) / max(n_seats, 1)
        logit = math.log(max(pick[i], 1e-6)) - CROWD_K * crowd
        if t == current:
            logit += KEEP_BONUS
        logits.append(logit)
    return softmax_logits(logits)


def draft_with_net(nets: Any, genomes: list[dict[str, Any]], rng: random.Random) -> dict[str, Any]:
    """TitanNet only: two opening rounds. No mid-match retitan."""
    from eveac_ai import TITAN_IDS as T

    n = len(genomes)
    lps1: list[Any] = []
    picks1: list[str] = []
    for g in genomes:
        obs = nets.titan_obs(g, census=None, current=None, round_i=1)
        i, lp = nets.sample_titan(obs)
        picks1.append(T[i])
        lps1.append(lp)
    counts = census(picks1)
    picks2: list[str] = []
    lps2: list[Any] = []
    for g, cur in zip(genomes, picks1):
        obs = nets.titan_obs(g, census=counts, current=cur, round_i=2)
        i, lp = nets.sample_titan(obs)
        picks2.append(T[i])
        lps2.append(lp)
    kept = sum(1 for a, b in zip(picks1, picks2) if a == b)
    return {
        "round1": picks1,
        "census": counts,
        "round2": picks2,
        "kept": kept,
        "n_seats": n,
        "titan_lps": [[a, b] for a, b in zip(lps1, lps2)],
        "backend": "titan_net",
    }


def draft_two_rounds(genomes: list[dict[str, Any]], rng: random.Random) -> dict[str, Any]:
    """Blind pick, publish counts, second pick (keep is always legal)."""
    n = len(genomes)
    picks1 = [_sample(titan_pick_vec(g), rng) for g in genomes]
    counts = census(picks1)
    picks2: list[str] = []
    for g, cur in zip(genomes, picks1):
        p2 = second_round_probs(titan_pick_vec(g), cur, counts, n)
        picks2.append(_sample(p2, rng))
    kept = sum(1 for a, b in zip(picks1, picks2) if a == b)
    return {
        "round1": picks1,
        "census": counts,
        "round2": picks2,
        "kept": kept,
        "n_seats": n,
    }


def draft_two_rounds_torch(genomes: list[dict[str, Any]], seed: int, device: object) -> dict[str, Any]:
    """Same two-round policy, batched on torch device (CUDA when available)."""
    import torch

    from eveac_ai import TITAN_IDS as T

    n = len(genomes)
    k = len(T)
    mat = torch.tensor([titan_pick_vec(g) for g in genomes], device=device, dtype=torch.float32)
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed) & 0x7FFFFFFF)
    idx1 = torch.multinomial(mat, 1, generator=gen).squeeze(-1)
    counts = torch.bincount(idx1, minlength=k).to(dtype=torch.float32)
    expected = n / float(k)
    crowd = (counts - expected) / max(n, 1)
    logp = torch.log(mat.clamp_min(1e-6))
    logits = logp - CROWD_K * crowd.unsqueeze(0)
    keep = torch.nn.functional.one_hot(idx1, k).to(dtype=torch.float32) * KEEP_BONUS
    logits = logits + keep
    logits = logits - logits.max(dim=-1, keepdim=True).values
    probs = torch.softmax(logits, dim=-1)
    idx2 = torch.multinomial(probs, 1, generator=gen).squeeze(-1)
    picks1 = [T[int(i)] for i in idx1.detach().cpu().tolist()]
    picks2 = [T[int(i)] for i in idx2.detach().cpu().tolist()]
    counts_d = {T[i]: int(counts[i].item()) for i in range(k)}
    kept = int((idx1 == idx2).sum().item())
    return {"round1": picks1, "census": counts_d, "round2": picks2, "kept": kept, "n_seats": n, "backend": "torch"}
