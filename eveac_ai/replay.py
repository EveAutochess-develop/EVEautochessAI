"""Replay text: opening snapshot + table results. No diagnostic dumps."""

from __future__ import annotations

from typing import Any

from eveac_ai.board_view import place_opening, render_half, render_legend, ship_label
from eveac_ai.capped_log import CappedLog


def income_for_round(round_i: int) -> int:
    if round_i <= 0:
        return 2
    if round_i == 1:
        return 3
    if round_i == 2:
        return 4
    return 5


def interest(gold: int) -> int:
    return min(5, gold // 10)


class ReplayWriter:
    def __init__(self, log: CappedLog, content: Any, board: dict[str, Any]) -> None:
        self.log = log
        self.content = content
        self.board = board

    def match_open(self, *, match_id: str, n_seats: int, draft: dict[str, Any], seats: list[dict[str, Any]]) -> None:
        lines = [
            f"对局复现 {match_id}  席数={n_seats}",
            "开场快照：两轮选泰坦后锁定；下列为各席本局泰坦与起始经济。",
            f"第1轮人数 {draft.get('census')}",
            f"第2轮人数见各席；维持 {draft.get('kept')}/{n_seats}",
            "",
        ]
        r1 = draft.get("round1") or []
        r2 = draft.get("round2") or []
        for s in seats:
            i = int(s["seat_id"])
            a = r1[i] if i < len(r1) else "?"
            b = r2[i] if i < len(r2) else s.get("titan")
            keep = "维持" if a == b else "改选"
            lines.append(
                f"席{i:02d} 泰坦 {a}→{b}（{keep}） 黄币={int(s.get('gold', 5))} 泰坦HP={s.get('titan_hp')}"
            )
        lines.append("")
        self.log.write("\n".join(lines))
        self.log.flush()

    def table_snapshot(
        self,
        *,
        round_i: int,
        seat_a: dict[str, Any],
        seat_b: dict[str, Any],
        pieces_a: list[dict[str, Any]],
        pieces_b: list[dict[str, Any]],
        gold_a0: int,
        gold_b0: int,
        result: dict[str, Any],
    ) -> None:
        ia, ib = int(seat_a["seat_id"]), int(seat_b["seat_id"])
        lines = [
            f"第{round_i + 1}轮 当场 席{ia:02d}({seat_a['titan']}) vs 席{ib:02d}({seat_b['titan']})",
            "",
            render_half(self.board, pieces_a, f"席{ia:02d} 半场"),
            render_legend(self.content, pieces_a),
            "",
            render_half(self.board, pieces_b, f"席{ib:02d} 半场"),
            render_legend(self.content, pieces_b),
            "",
            (
                f"结果 平局  双方泰坦HP 席{ia:02d}={seat_a['titan_hp']:.0f} 席{ib:02d}={seat_b['titan_hp']:.0f}"
                if result.get("draw")
                else (
                    f"结果 胜=席{int(result['winner']):02d}  败=席{int(result['loser']):02d}  "
                    f"败方泰坦HP {result['loser_hp']:.0f}{' 出局' if result.get('eliminated') else ''}"
                )
            ),
            f"经济 席{ia:02d} {gold_a0}→{int(seat_a['gold'])}  席{ib:02d} {gold_b0}→{int(seat_b['gold'])}  "
            f"（收入/利息/胜负已计入）",
            "",
        ]
        self.log.write("\n".join(lines))
        self.log.flush()

    def pve_snapshot(
        self,
        *,
        round_i: int,
        seat: dict[str, Any],
        pieces: list[dict[str, Any]],
        gold0: int,
        task: str,
        won: bool,
        freighter_id: str = "",
        creep_ids: list[str] | None = None,
    ) -> None:
        """Same layout contract as PVP tables: Field+Hangar numbers + legend."""
        sid = int(seat["seat_id"])
        task_zh = {"pve_eliminate": "消灭", "pve_salvage": "抢救"}.get(str(task), str(task))
        creeps = "、".join(str(x) for x in (creep_ids or [])[:12]) or "—"
        if freighter_id:
            creeps = f"{creeps}；货舰={freighter_id}"
        lines = [
            f"第{round_i + 1}轮 PVE·{task_zh} 席{sid:02d}({seat.get('titan')})  "
            f"{'成功' if won else '失败'}  泰坦HP={float(seat.get('titan_hp') or 0):.0f}",
            f"野怪编队 {creeps}",
            "",
            render_half(self.board, pieces, f"席{sid:02d} 半场（备战落子）"),
            render_legend(self.content, pieces),
            "",
            f"经济 席{sid:02d} {int(gold0)}→{int(seat.get('gold') or 0)}  （采矿/击毁/胜场包已计入）",
            "",
        ]
        self.log.write("\n".join(lines))
        self.log.flush()

    def round_marker(self, *, match_id: str, round_i: int, kind: str, alive: int) -> None:
        self.log.write(f"—— {match_id} 第{round_i + 1}轮结束 kind={kind} 存活={alive} ——\n")
        self.log.flush()

    def ranking(self, ranked: list[dict[str, Any]], keep: int) -> None:
        lines = ["终局名次"]
        for i, s in enumerate(ranked):
            mark = "精英" if i < keep else ""
            lines.append(
                f"  #{i + 1} 席{int(s['seat_id']):02d} {s['titan']} "
                f"HP={s['titan_hp']:.0f} W={s['wins']} L={s['losses']} "
                f"黄={int(s.get('gold', 0))} {'在场' if s['alive'] else '出局'} {mark}"
            )
        lines.append("")
        self.log.write("\n".join(lines))
        self.log.flush()
