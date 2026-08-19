"""Apply function-module effects onto SimShip at spawn. Periodic repair uses repair dict."""

from __future__ import annotations

from typing import Any

from eveac_ai.ship import SimShip


def _f(v: Any, d: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def apply_equips(ship: SimShip, content: Any, equip_ids: list[str]) -> None:
    for raw in (equip_ids or [])[:3]:
        eid = str(raw).split(":", 1)[0]
        meta = content.equip_meta.get(eid) or {}
        for e in meta.get("effects") or []:
            if not isinstance(e, dict):
                continue
            op = str(e.get("op") or "")
            stat = str(e.get("stat") or "")
            if op == "mul_stat" or op == "mul_stat_active":
                mul = _f(e.get("mul"), 1.0)
                if stat == "speed":
                    ship.speed *= mul
                elif stat in ("damage",):
                    ship.damage = {k: v * mul for k, v in ship.damage.items()}
                elif stat == "scan_resolution":
                    ship.scan *= mul
            elif op == "add_stat":
                add = _f(e.get("add"), _f(e.get("value")))
                if stat in ("shield_hp",):
                    ship.max_shield += add
                    ship.shield += add
                elif stat in ("armor_hp",):
                    ship.max_armor += add
                    ship.armor += add
                elif stat in ("structure_hp",):
                    ship.max_structure += add
                    ship.structure += add
                elif stat == "capacitor_capacity":
                    ship.cap_capacity += add
                    ship.cap_current += add
            elif op == "add_resist" or op == "add_resist_active":
                add = _f(e.get("add"), _f(e.get("value"))) / 100.0
                layer = str(e.get("layer") or "armor")
                r = ship.armor_resist if layer == "armor" else ship.shield_resist
                for k in r:
                    r[k] = min(0.95, r[k] + add)
            elif op == "repair":
                ship.repair["shield"] += _f(e.get("shield"))
                ship.repair["armor"] += _f(e.get("armor"))
                ship.repair["structure"] += _f(e.get("structure"))
            elif op == "mul_damage_gate":
                ship.damage = {k: v * _f(e.get("mul"), 1.0) for k, v in ship.damage.items()}
        if eid == "mixed_lance" or str(meta.get("activate") or "") == "mixed_lance":
            if ship.capital_role == "dreadnought" or ship.ship_group == "dreadnought":
                ship.has_lance = True
        if str(meta.get("shop_category")) == "repair":
            # JSON repair rows often only in effects; if empty, small fallback from size
            if sum(ship.repair.values()) <= 0:
                sz = str(meta.get("size") or "S")
                ship.repair["armor"] += {"S": 69, "M": 276, "L": 690, "XL": 960}.get(sz, 40)
