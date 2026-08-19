"""Deterministic farm RNG. Not bit-identical to Godot RandomNumberGenerator."""

from __future__ import annotations

import random


class FarmRng:
    def __init__(self, seed: int) -> None:
        self._r = random.Random(seed if seed else 1)

    def randf(self) -> float:
        return self._r.random()

    def randi_range(self, lo: int, hi: int) -> int:
        if hi < lo:
            lo, hi = hi, lo
        return self._r.randint(lo, hi)
