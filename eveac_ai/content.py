from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path | None = None) -> dict[str, Any]:
    p = path or (ROOT / "config.json")
    return json.loads(p.read_text(encoding="utf-8"))


def data_dir(cfg: dict[str, Any] | None = None) -> Path:
    cfg = cfg or load_config()
    return (ROOT / str(cfg["game_data_dir"])).resolve()


def content_rev(d: Path) -> str:
    ships = d / "ships"
    n = len(list(ships.glob("*.json"))) if ships.is_dir() else 0
    return f"ships:{n}"


def load_economy(d: Path) -> dict[str, Any]:
    p = d / "balance" / "economy.json"
    return load_json(p) if p.is_file() else {}


def load_fetters(d: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    folder = d / "fetters"
    if not folder.is_dir():
        return out
    for p in folder.glob("*.json"):
        blob = load_json(p)
        if isinstance(blob, dict) and blob.get("id"):
            out[str(blob["id"])] = blob
    return out


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_combat(d: Path) -> dict[str, Any]:
    p = d / "balance" / "combat.json"
    return load_json(p) if p.is_file() else {}


def load_board(d: Path) -> dict[str, Any]:
    p = d / "balance" / "board.json"
    return load_json(p) if p.is_file() else {}


def load_titan_pvp(d: Path) -> dict[str, Any]:
    p = d / "balance" / "titan_pvp.json"
    return load_json(p) if p.is_file() else {}


def titan_max_hp(titan_pvp: dict | None = None) -> float:
    """Shield+armor+structure (MULTIPLAYER_PVP §2.4). Default 100+100+100."""
    tp = titan_pvp or {}
    return (
        float(tp.get("pipe_shield_max") or 100)
        + float(tp.get("pipe_armor_max") or 100)
        + float(tp.get("pipe_structure_max") or 100)
    )


def titan_pvp_loss(titan_pvp: dict | None = None) -> float:
    return float((titan_pvp or {}).get("pvp_loss_damage") or 20)


def remaining_pvp_losses(hp: float, titan_pvp: dict | None = None) -> int:
    """Ops reads HP as remaining PVP defeats, not leftover pool."""
    loss = max(1.0, titan_pvp_loss(titan_pvp))
    return max(0, int(float(hp) // loss))


def load_match_flow(d: Path) -> dict[str, Any]:
    p = d / "balance" / "match_flow.json"
    return load_json(p) if p.is_file() else {"sim_fixed_step_s": 0.05, "battle_duration_s": 900}


def load_equip_ids(d: Path) -> list[str]:
    path = d / "equipment" / "function_modules.json"
    if not path.is_file():
        return []
    items = (load_json(path).get("items") or {})
    return [str(k) for k in items.keys()]


def load_equip_meta(d: Path) -> dict[str, dict[str, Any]]:
    path = d / "equipment" / "function_modules.json"
    if not path.is_file():
        return {}
    items = load_json(path).get("items") or {}
    return {str(k): v for k, v in items.items() if isinstance(v, dict)}


def load_modules(d: Path) -> dict[str, dict[str, Any]]:
    path = d / "equipment" / "modules.json"
    if not path.is_file():
        return {}
    blob = load_json(path)
    if not isinstance(blob, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for k, v in blob.items():
        if isinstance(v, dict):
            out[str(k)] = v
    return out


def load_ships(d: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for folder in ("ships", "unmanned_units"):
        root = d / folder
        if not root.is_dir():
            continue
        for p in sorted(root.glob("*.json")):
            blob = load_json(p)
            if not isinstance(blob, dict):
                continue
            try:
                sid = str(int(float(blob.get("id", p.stem))))
            except (TypeError, ValueError):
                sid = p.stem
            out[sid] = blob
    return out


def star_at(ship: dict[str, Any], star: int = 1) -> dict[str, Any]:
    stars = ship.get("stars") or []
    if not stars:
        return {}
    if star < 1:
        return {}
    if star <= len(stars):
        row = stars[star - 1]
        return row if isinstance(row, dict) else {}
    if ship.get("is_unmanned") and isinstance(stars[0], dict):
        from eveac_ai.weapon_derive import synthesize_unmanned_star

        return synthesize_unmanned_star(stars[0], star)
    return {}


class Content:
    def __init__(self, d: Path | None = None, cfg: dict[str, Any] | None = None) -> None:
        self.cfg = cfg or load_config()
        self.dir = d or data_dir(self.cfg)
        self.combat = load_combat(self.dir)
        self.match_flow = load_match_flow(self.dir)
        self.ships = load_ships(self.dir)
        self.modules = load_modules(self.dir)
        self.equip_ids = load_equip_ids(self.dir)
        self.equip_meta = load_equip_meta(self.dir)
        self.board = load_board(self.dir)
        self.economy = load_economy(self.dir)
        self.titan_pvp = load_titan_pvp(self.dir)
        self.fetters = load_fetters(self.dir)
        self.rev = content_rev(self.dir)

    def f(self, key: str, default: float) -> float:
        try:
            return float(self.combat.get(key, default))
        except (TypeError, ValueError):
            return default
