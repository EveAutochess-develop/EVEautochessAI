import json
from pathlib import Path

import pytest

from eveac_ai.content import Content
from eveac_ai.gpu_kernel import GpuBattleKernel


def _jobs(n: int) -> list[dict]:
    fixture = json.loads((Path(__file__).parent / "golden" / "1v1.json").read_text(encoding="utf-8"))
    return [
        {
            "fleet_a": fixture["fleet_a"],
            "fleet_b": fixture["fleet_b"],
            "seed": int(fixture["seed"]) + i,
            "match_id": f"slot-{i}",
            "round_i": 0,
            "seat_a": 0,
            "seat_b": 1,
        }
        for i in range(n)
    ]


def test_fight_batch_slots_return_all_jobs_in_order():
    torch = pytest.importorskip("torch")
    k = GpuBattleKernel(Content(), torch.device("cpu"), sparse_dt=True)
    jobs = _jobs(3)
    packs = k.fight_batch(jobs, slots=2)
    assert [p["match_id"] for p in packs] == ["slot-0", "slot-1", "slot-2"]
    assert all(len(p.get("seats") or []) == 2 for p in packs)
    timing = k.last_timing or {}
    assert timing.get("slotted") is True
    assert int(timing.get("slots") or 0) == 2
    assert int(timing.get("n_jobs") or 0) == 3
    assert int(timing.get("n_refill") or 0) >= 1
    assert len(packs) == 3
    assert "finishes" in timing
    assert len(timing["finishes"]) == 3
    assert float(timing.get("wall_s") or 0) > 0


def test_fight_batch_unslotted_same_count():
    torch = pytest.importorskip("torch")
    k = GpuBattleKernel(Content(), torch.device("cpu"), sparse_dt=True)
    jobs = _jobs(2)
    packs = k.fight_batch(jobs, slots=0)
    assert len(packs) == 2
    timing = k.last_timing or {}
    assert timing.get("slotted") is False
    assert int(timing.get("n_refill") or 0) == 0
