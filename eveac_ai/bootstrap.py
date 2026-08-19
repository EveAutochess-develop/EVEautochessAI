"""Handbook-aligned first-batch priors (AI_SELFPLAY §9). Written at implement time, not a runtime LLM call."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from eveac_ai import SCHEMA_VER, STANCE_FLOOR, STANCE_IDS, TITAN_IDS
from eveac_ai.content import Content, star_at
from eveac_ai.prepare import is_shop_combat_hull

ROOT = Path(__file__).resolve().parents[1]


def _clamp(v: float, lo: float = 0.05, hi: float = 0.99) -> float:
    return max(lo, min(hi, v))


def ship_weight(hull: dict[str, Any], titan: str) -> float:
    """Relative pick weight for one titan slice. Keys are ids; no Chinese names."""
    race = str(hull.get("race", "")).lower()
    w = 0.42
    if race == titan:
        w += 0.32
    elif titan == "angel" and race in ("minmatar", "gallente", "angel"):
        w += 0.18
    elif race in TITAN_IDS:
        w += 0.04
    if hull.get("is_logistic") or (star_at(hull, 1).get("is_logistic")):
        w += 0.12
    if hull.get("is_unmanned"):
        w -= 0.20
    if hull.get("is_mining_ship"):
        w += 0.28
    if str(hull.get("capital_role", "")).lower() == "covert_cyno":
        w += 0.10
    if hull.get("requires_cyno_entry"):
        w += 0.08
    cost = float(hull.get("cost") or 0)
    if cost <= 3:
        w += 0.06
    elif cost >= 22:
        w -= 0.08
    fet = hull.get("fetter_ids") or []
    if titan in [str(x).lower() for x in fet]:
        w += 0.08
    dmg = star_at(hull, 1).get("damage") if isinstance(star_at(hull, 1).get("damage"), dict) else {}
    dsum = sum(float(dmg.get(k) or 0) for k in ("emp", "thermal", "kinetic", "explosive"))
    if dsum > 1:
        w += 0.05
    return round(_clamp(w), 4)


def equip_weight(eid: str, meta: dict[str, Any]) -> float:
    blob = str(eid).lower() + " " + str(meta.get("id", "")).lower() + " " + str(meta.get("line", "")).lower()
    w = 0.45
    if any(x in blob for x in ("repair", "remote", "armor_rep", "shield_boost", "logistics")):
        w = 0.72
    elif any(x in blob for x in ("afterburner", "mwd", "microwarp", "ab_")):
        w = 0.62
    elif any(x in blob for x in ("web", "scram", "disrupt", "paint", "damp")):
        w = 0.58
    elif any(x in blob for x in ("shield", "resist", "harden")):
        w = 0.55
    return round(_clamp(w), 4)


def build_genome(content: Content) -> dict[str, Any]:
    slices: dict[str, Any] = {}
    shop_ships = {sid: hull for sid, hull in content.ships.items() if is_shop_combat_hull(hull)}
    for titan in TITAN_IDS:
        slices[titan] = {
            "ship": {sid: ship_weight(hull, titan) for sid, hull in shop_ships.items()},
            "equip": {eid: equip_weight(eid, content.equip_meta.get(eid, {})) for eid in content.equip_ids},
        }
    stance = {
        "economy": 0.22,
        "offense": 0.26,
        "logistics": 0.20,
        "speed_control": 0.16,
        "formation": 0.16,
    }
    ssum = sum(stance.values())
    stance = {k: round(max(STANCE_FLOOR, v / ssum), 4) for k, v in stance.items()}
    t = sum(stance.values())
    stance = {k: round(v / t, 4) for k, v in stance.items()}
    n_t = float(len(TITAN_IDS))
    titan_pick = {t: round(1.0 / n_t, 4) for t in TITAN_IDS}
    return {
        "schema_ver": SCHEMA_VER,
        "content_rev": content.rev,
        "origin": "llm_bootstrap",
        "note": "First-batch priors from AI_SELFPLAY + shop-combat ship/equip tables only (no titans/wrecks). Not shop seed. titan_pick is policy, not a brand.",
        "titan_ids": list(TITAN_IDS),
        "titan_slices": slices,
        "titan_pick": titan_pick,
        "stance": stance,
        "stance_floor": STANCE_FLOOR,
        "prepare": {
            "buy_n": 4,
            "fit_slots": 2,
            "hangar_keep_if_economy_ge": 0.22,
        },
        "black_box": {"bootstrap": "handbook_race_logi_cost"},
    }


def write_prior(path: Path | None = None) -> Path:
    content = Content()
    genome = build_genome(content)
    out = path or (ROOT / "priors" / "llm_bootstrap.genome.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(genome, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


if __name__ == "__main__":
    p = write_prior()
    print(p)
