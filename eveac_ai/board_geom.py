"""Board cell → sim XZ in cells (Godot BoardController.cell_to_world / wu)."""

from __future__ import annotations

from typing import Any


def field_cols_at(board: dict[str, Any], z: int) -> int:
    fw = int(board.get("field_width") or 12)
    extra = int(board.get("field_odd_row_extra") or 1)
    if int(z) % 2 == 1:
        return fw + extra
    return fw


def field_cell_xz(
    board: dict[str, Any],
    team: int,
    x: int,
    z: int,
    *,
    world_units_per_cell: float = 3.0,
) -> tuple[float, float]:
    """Mirror Godot field placement; return coordinates in combat cell units."""
    ox = float(board.get("hex_offset_x") if board.get("hex_offset_x") is not None else -3.0)
    oz = float(board.get("hex_offset_z") if board.get("hex_offset_z") is not None else -2.5)
    fh = int(board.get("field_height") or 6)
    gap = float(board.get("center_gap_z") if board.get("center_gap_z") is not None else 4.0)
    hoz = abs(oz)
    origin_x = 0.0
    origin_z = float(fh - 1) * hoz + gap
    if team == 1:
        if "ai_origin_x" in board:
            origin_x = float(board.get("ai_origin_x") or 0.0)
        if "ai_origin_z" in board:
            origin_z = float(board.get("ai_origin_z") or 0.0)
        else:
            origin_z = -origin_z
    else:
        if "player_origin_x" in board:
            origin_x = float(board.get("player_origin_x") or 0.0)
        if "player_origin_z" in board:
            origin_z = float(board.get("player_origin_z") or 0.0)
    cols = field_cols_at(board, z)
    row_left = float(cols - 1) * abs(ox) * 0.5
    offset_x = row_left + float(x) * ox
    offset_z = float(z) * oz
    if team == 1:
        offset_x = -offset_x
        offset_z = -offset_z
    wu = max(float(world_units_per_cell) or 3.0, 1e-6)
    return (origin_x + offset_x) / wu, (origin_z + offset_z) / wu


def play_bounds_cells(
    board: dict[str, Any],
    *,
    world_units_per_cell: float = 3.0,
    margin_cells: float = 0.75,
) -> tuple[float, float, float, float]:
    """(min_x, max_x, min_z, max_z) in cell units covering both team fields."""
    xs: list[float] = []
    zs: list[float] = []
    fh = int(board.get("field_height") or 6)
    for team in (0, 1):
        for z in (0, max(0, fh - 1)):
            cols = field_cols_at(board, z)
            for x in (0, max(0, cols - 1)):
                cx, cz = field_cell_xz(board, team, x, z, world_units_per_cell=world_units_per_cell)
                xs.append(cx)
                zs.append(cz)
    if not xs:
        return -8.0, 8.0, -8.0, 8.0
    return min(xs) - margin_cells, max(xs) + margin_cells, min(zs) - margin_cells, max(zs) + margin_cells


def clamp_xz(x: float, z: float, bounds: tuple[float, float, float, float]) -> tuple[float, float]:
    return max(bounds[0], min(bounds[1], x)), max(bounds[2], min(bounds[3], z))
