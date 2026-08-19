"""Half-board numbered tables for replay text. Sizes come from board.json at runtime."""

from __future__ import annotations

from typing import Any


def cols_for_row(board: dict[str, Any], z: int) -> int:
    w = int(board.get("field_width") or 8)
    extra = int(board.get("field_odd_row_extra") or 0)
    return w + (extra if z % 2 else 0)


def hangar_n(board: dict[str, Any]) -> int:
    return int(board.get("hangar_width") or 8)


def field_h(board: dict[str, Any]) -> int:
    return int(board.get("field_height") or 6)


def ship_label(content: Any, ship_id: str) -> str:
    hull = (content.ships if content else {}).get(str(ship_id)) or {}
    name = str(hull.get("name") or "")
    return f"{ship_id} {name}".strip()


def place_opening(fleet: list[str], board: dict[str, Any]) -> list[dict[str, Any]]:
    """Put leading ships on field row 0; overflow then remainder into hangar. star=1, no fit yet."""
    h = hangar_n(board)
    c0 = cols_for_row(board, 0)
    pieces: list[dict[str, Any]] = []
    tok = 1
    field_n = min(len(fleet), c0)
    for i, sid in enumerate(fleet[:field_n]):
        pieces.append({"token": tok, "ship_id": str(sid), "star": 1, "equips": [], "slot": "field", "x": i, "z": 0})
        tok += 1
    hangar_ships = fleet[field_n:]
    for i, sid in enumerate(hangar_ships[:h]):
        pieces.append({"token": tok, "ship_id": str(sid), "star": 1, "equips": [], "slot": "hangar", "x": i, "z": 0})
        tok += 1
    return pieces


def render_half(board: dict[str, Any], pieces: list[dict[str, Any]], title: str) -> str:
    by_field = {(p["x"], p["z"]): p["token"] for p in pieces if p["slot"] == "field"}
    by_hang = {p["x"]: p["token"] for p in pieces if p["slot"] == "hangar"}
    hgt = field_h(board)
    hn = hangar_n(board)
    lines = [title]
    lines.append("Field（近隔离带在上，近等候席在下；格内为编号）")
    for z in range(hgt - 1, -1, -1):
        cols = cols_for_row(board, z)
        cells = []
        for x in range(cols):
            t = by_field.get((x, z))
            cells.append(f"{t:>3}" if t is not None else "  .")
        lines.append(f" z{z} " + " ".join(cells))
    hang_cells = []
    for x in range(hn):
        t = by_hang.get(x)
        hang_cells.append(f"{t:>3}" if t is not None else "  .")
    lines.append("Hangar（等候席）")
    lines.append("    " + " ".join(hang_cells))
    return "\n".join(lines)


def render_legend(content: Any, pieces: list[dict[str, Any]]) -> str:
    if not pieces:
        return "注释：本半场无舰。"
    lines = ["注释"]
    for p in sorted(pieces, key=lambda x: int(x["token"])):
        eq = p.get("equips") or []
        eq_s = "、".join(str(e) for e in eq) if eq else "无"
        where = "等候席" if p["slot"] == "hangar" else f"场上 z{p['z']} x{p['x']}"
        lines.append(
            f"  #{p['token']} {ship_label(content, str(p['ship_id']))} ★{int(p.get('star') or 1)} "
            f"装:{eq_s}  {where}"
        )
    return "\n".join(lines)
