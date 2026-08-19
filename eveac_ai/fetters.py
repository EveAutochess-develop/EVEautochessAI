"""Field fetters + titan meta. Hangar does not count."""

from __future__ import annotations

from typing import Any

from eveac_ai.ship import SimShip


def _pct(val: float, step: int) -> float:
    return (val + step) / 100.0


def apply_fetters(ships: list[SimShip], content: Any, titan_by_team: dict[int, str]) -> None:
    tables = content.fetters or {}
    for team in (0, 1):
        field = [s for s in ships if s.team == team and s.alive() and not getattr(s, "hold_until", 0)]
        counts: dict[str, int] = {}
        for s in field:
            hull = content.ships.get(s.ship_id) or {}
            for fid in hull.get("fetter_ids") or []:
                counts[str(fid)] = counts.get(str(fid), 0) + 1
        active: list[tuple[dict[str, Any], int, int]] = []
        for fid, n in counts.items():
            blob = tables.get(fid) or {}
            if blob.get("meta"):
                continue
            effects = [e for e in (blob.get("effects") or []) if isinstance(e, dict)]
            if not effects:
                continue
            needs = [int(e.get("champion_count") or 0) for e in effects if int(e.get("champion_count") or 0) > 0]
            base_need = min(needs) if needs else 0
            eligible = [e for e in effects if int(e.get("champion_count") or 0) <= n]
            if not eligible:
                continue
            pick = max(eligible, key=lambda e: int(e.get("champion_count") or 0))
            step = max(0, n - base_need) if base_need > 0 else 0
            active.append((pick, step, n))
        titan = str(titan_by_team.get(team) or "")
        tkey = titan if titan.startswith("titan_") else f"titan_{titan}"
        tblob = tables.get(tkey) or {}
        if tblob.get("meta"):
            for e in tblob.get("effects") or []:
                if str(e.get("effect_type")) == "ShopRaceWeight":
                    continue
                active.append((e, 0, 0))
        for s in field:
            hull = content.ships.get(s.ship_id) or {}
            fids = {str(x) for x in (hull.get("fetter_ids") or [])}
            for e, step, _n in active:
                tgt = str(e.get("effect_target") or "SelfAll")
                et = str(e.get("effect_type") or "")
                vt = str(e.get("effect_value_type") or "Percentage")
                val = float(e.get("value") or 0)
                if tgt == "SelfFetter" and fids and et:
                    pass
                elif tgt == "SelfFetter":
                    continue
                if vt == "Percentage":
                    mul = 1.0 + _pct(val, step)
                else:
                    mul = 1.0
                if et in ("Damage",):
                    s.damage = {k: v * mul for k, v in s.damage.items()}
                elif et in ("ArmorHP",):
                    s.max_armor *= mul
                    s.armor *= mul
                elif et in ("ShieldHP",):
                    s.max_shield *= mul
                    s.shield *= mul
                elif et in ("FlatHP",):
                    extra = val * (1.0 + 0.01 * step)
                    s.max_structure += extra
                    s.structure += extra
                elif et in ("AttackSpeed",):
                    s.attack_duration = s.attack_duration / mul
                elif et in ("Speed",):
                    s.speed *= mul
                elif et in ("ArmorResist", "ShieldResist"):
                    key = "armor_resist" if "Armor" in et else "shield_resist"
                    r = getattr(s, key)
                    setattr(s, key, {k: min(0.95, v + (val + step) / 100.0) for k, v in r.items()})
                elif et in ("RemoteRepair", "Repair", "ArmorHeal", "ShieldHeal"):
                    s.repair = {k: v * mul for k, v in s.repair.items()}
