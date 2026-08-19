"""Ring buffer of past elite genomes for frozen-seat opponents (~15%)."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


def pool_dir(samples: Path) -> Path:
    d = samples / "elite_pool"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_generation(samples: Path, gen_i: int, genome: dict[str, Any], *, keep: int = 8) -> Path:
    d = pool_dir(samples)
    path = d / f"gen{int(gen_i)}.genome.json"
    path.write_text(json.dumps(genome, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    files = sorted(d.glob("gen*.genome.json"), key=lambda p: p.stat().st_mtime)
    while len(files) > max(1, int(keep)):
        old = files.pop(0)
        try:
            old.unlink()
        except OSError:
            pass
    return path


def list_pool(samples: Path) -> list[Path]:
    d = samples / "elite_pool"
    if not d.is_dir():
        return []
    return sorted(d.glob("gen*.genome.json"), key=lambda p: p.stat().st_mtime)


def draw_frozen_genomes(samples: Path, n: int, rng: random.Random) -> list[dict[str, Any]]:
    files = list_pool(samples)
    if n <= 0 or not files:
        return []
    out: list[dict[str, Any]] = []
    for _ in range(n):
        path = rng.choice(files)
        try:
            g = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(g, dict):
            out.append(g)
    return out
