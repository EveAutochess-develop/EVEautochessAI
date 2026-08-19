from pathlib import Path

from eveac_ai.board_view import place_opening, render_half, render_legend
from eveac_ai.capped_log import CappedLog


def test_capped_log_rotates_under_tiny_cap(tmp_path: Path):
    p = tmp_path / "diag.log"
    log = CappedLog(p, cap=80, kind="diag")
    log.write("x" * 40)
    log.write("y" * 50)
    log.close()
    text = p.read_text(encoding="utf-8")
    assert "rotated" in text
    assert "y" * 20 in text
    assert p.stat().st_size <= 80 + 40


def test_replay_table_uses_tokens_and_legend():
    board = {"field_width": 4, "field_odd_row_extra": 1, "field_height": 2, "hangar_width": 3}
    pieces = place_opening(["4", "6", "11", "13", "15"], board)
    assert any(p["slot"] == "field" for p in pieces)
    assert any(p["slot"] == "hangar" for p in pieces)
    table = render_half(board, pieces, "席00 半场")
    assert "Hangar" in table
    assert "Field" in table
    for p in pieces:
        assert f"{p['token']:>3}" in table or str(p["token"]) in table

    class C:
        ships = {"4": {"name": "富豪级"}, "6": {"name": "龙骑兵级"}, "11": {"name": "小鹰级"}, "13": {"name": "秃鹫级"}, "15": {"name": "海燕级"}}

    legend = render_legend(C(), pieces)
    assert "#1" in legend
    assert "★1" in legend
    assert "装:无" in legend
