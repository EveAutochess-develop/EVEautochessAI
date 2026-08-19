"""Placement MLP: ship+equips in, legal cell out. Combat reward backprops (REINFORCE)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from eveac_ai.content import star_at
from eveac_ai.ship import damage_sum

FEAT = 24
HID = 64
MAX_CELLS = 64
LR = 3e-4


def _is_cyno(hull: dict[str, Any] | None) -> bool:
    return str((hull or {}).get("capital_role", "")).lower() == "covert_cyno"


def _is_flag(hull: dict[str, Any] | None) -> bool:
    return bool((hull or {}).get("requires_cyno_entry"))


def _cells(board: dict[str, Any]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for z in range(int(board.get("field_height") or 6)):
        w = int(board.get("field_width") or 8) + (int(board.get("field_odd_row_extra") or 0) if z % 2 else 0)
        for x in range(w):
            out.append((x, z))
    return out[:MAX_CELLS]


def _f(v: Any, d: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def ship_equip_feat(content: Any, ship_id: str, equips: list[str], stance: dict[str, Any]) -> list[float]:
    hull = content.ships.get(str(ship_id)) or {}
    st = star_at(hull, 1)
    dmg = st.get("damage") if isinstance(st.get("damage"), dict) else {}
    dsum = damage_sum({k: _f(dmg.get(k)) for k in ("emp", "thermal", "kinetic", "explosive")})
    eq_cost = 0.0
    for raw in (equips or [])[:3]:
        eid = str(raw).split(":", 1)[0]
        eq_cost += _f((content.equip_meta.get(eid) or {}).get("cost"))
    vec = [
        _f(hull.get("cost")) / 30.0,
        _f(st.get("shield_hp")) / 4000.0,
        _f(st.get("armor_hp")) / 4000.0,
        _f(st.get("structure_hp")) / 4000.0,
        _f(hull.get("speed")) / 400.0,
        _f(hull.get("signature_radius")) / 400.0,
        1.0 if (hull.get("is_logistic") or st.get("is_logistic")) else 0.0,
        1.0 if _is_cyno(hull) else 0.0,
        1.0 if _is_flag(hull) else 0.0,
        dsum / 400.0,
        _f(hull.get("attack_cycle_s")) / 10.0,
        eq_cost / 40.0,
        float(len(equips or [])) / 3.0,
        float(stance.get("offense") or 0.0),
        float(stance.get("formation") or 0.0),
        float(stance.get("speed_control") or 0.0),
        float(stance.get("logistics") or 0.0),
        float(stance.get("economy") or 0.0),
        _f(hull.get("optimal") or (st.get("optimal") if isinstance(st, dict) else 0)) / 40.0,
        _f(hull.get("capacitor_capacity")) / 4000.0,
        1.0,
        1.0,
        1.0,
        1.0,
    ]
    return (vec + [0.0] * FEAT)[:FEAT]


class PlaceNet:
    def __init__(self, device: object | None, path: Path | None = None) -> None:
        import torch
        import torch.nn as nn

        self.torch = torch
        self.device = device or torch.device("cpu")
        self.path = path
        self.net = nn.Sequential(
            nn.Linear(FEAT + MAX_CELLS, HID),
            nn.Tanh(),
            nn.Linear(HID, HID),
            nn.Tanh(),
            nn.Linear(HID, MAX_CELLS),
        ).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=LR)
        self._pending: list[tuple[Any, float]] = []
        if path and path.is_file():
            blob = torch.load(path, map_location=self.device)
            self.net.load_state_dict(blob)

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.torch.save(self.net.state_dict(), self.path)

    def place_field(
        self,
        content: Any,
        board: dict[str, Any],
        stance: dict[str, Any],
        field_pieces: list[dict[str, Any]],
    ) -> None:
        torch = self.torch
        cells = _cells(board)
        n = len(cells)
        occ = [0.0] * MAX_CELLS
        taken: set[tuple[int, int]] = set()
        self.net.train()
        for p in field_pieces:
            feat = ship_equip_feat(content, p["ship_id"], p.get("equips") or [], stance)
            x = torch.tensor(feat + occ, dtype=torch.float32, device=self.device).unsqueeze(0)
            logits = self.net(x)[0, :n]
            mask = torch.zeros(n, device=self.device)
            for i, xz in enumerate(cells):
                if xz in taken:
                    mask[i] = -1e9
            dist = torch.distributions.Categorical(logits=logits + mask)
            idx = int(dist.sample().item())
            lp = dist.log_prob(torch.tensor(idx, device=self.device))
            p["x"], p["z"] = cells[idx]
            p["_place_lp"] = lp
            taken.add(cells[idx])
            occ[idx] = 1.0

    def remember_pieces(self, pieces: list[dict[str, Any]], reward: float) -> None:
        for p in pieces:
            lp = p.pop("_place_lp", None)
            if lp is not None:
                self._pending.append((lp, float(reward)))

    def backward_step(self) -> float:
        if not self._pending:
            return 0.0
        torch = self.torch
        loss = torch.zeros((), device=self.device)
        for lp, r in self._pending:
            loss = loss - lp * r
        loss = loss / max(1, len(self._pending))
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self._pending.clear()
        return float(loss.detach().item())
