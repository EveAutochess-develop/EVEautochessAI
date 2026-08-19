"""BattleKernel stub. backend cpu_stub | gpu_stub; both empty-legal this round."""

from __future__ import annotations

import hashlib
from typing import Any


ALLOWED_BACKENDS = ("cpu_stub", "gpu_stub")


def _score(genome: dict[str, Any], titan: str) -> float:
    stance = genome.get("stance") or {}
    offense = float(stance.get("offense", 0.2))
    logistics = float(stance.get("logistics", 0.2))
    slice_ = (genome.get("titan_slices") or {}).get(titan) or {}
    ships = slice_.get("ship") or {}
    mass = sum(float(v) for v in ships.values()) / max(1, len(ships))
    return offense * 1.2 + logistics * 0.3 + mass * 0.01


def fight(
    *,
    backend: str,
    match_id: str,
    round_i: int,
    seat_a: dict[str, Any],
    seat_b: dict[str, Any],
) -> dict[str, Any]:
    if backend not in ALLOWED_BACKENDS:
        raise ValueError(f"unsupported backend {backend!r}; stub allows {ALLOWED_BACKENDS}")
    sa = _score(seat_a["genome"], seat_a["titan"])
    sb = _score(seat_b["genome"], seat_b["titan"])
    # Stable jitter so identical genomes still break ties without a real kernel.
    jitter = int(hashlib.sha256(f"{match_id}:{round_i}:{seat_a['seat_id']}:{seat_b['seat_id']}".encode()).hexdigest()[:8], 16)
    sa += (jitter % 1000) / 1e6
    a_wins = sa >= sb
    return {
        "schema_ver": "1",
        "backend": backend,
        "match_id": match_id,
        "round": round_i,
        "seats": [
            _seat_pack(seat_a, won=a_wins, hint=sa),
            _seat_pack(seat_b, won=not a_wins, hint=sb),
        ],
    }


def _seat_pack(seat: dict[str, Any], *, won: bool, hint: float) -> dict[str, Any]:
    titan_hp = 80.0 if won else 60.0
    return {
        "seat_id": int(seat["seat_id"]),
        "won": won,
        "titan_hp": titan_hp,
        "loss_comp_gold": 0 if won else 1,
        "rank_hint": float(hint),
        "ships": [],
        "kill_calendar": [],
        "traj_snapshots": [],
        "hp_trends": [],
        "lock_timeline": [],
    }
