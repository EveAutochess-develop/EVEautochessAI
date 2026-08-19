"""Run a single table (2 seats), write a readable log and ship-weight deltas from telemetry."""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path

from eveac_ai.content import Content, load_config
from eveac_ai.kernel import BattleKernel
from eveac_ai.priors import derive_seat_genome, load_bootstrap, pick_fleet
from eveac_ai.prepare import is_shop_combat_hull

ROOT = Path(__file__).resolve().parents[1]


def _names(content: Content, sid: str) -> str:
    hull = content.ships.get(str(sid)) or {}
    return f"{sid} {hull.get('name', '')}".strip()


def apply_telemetry_delta(
    genome: dict,
    titan: str,
    seat_row: dict,
    won: bool,
    content: Content | None = None,
) -> list[tuple[str, float, float]]:
    """Slow local update: ships that dealt more than they took get a bump; reverse for sponges that dealt 0."""
    sl = (genome.get("titan_slices") or {}).get(titan) or {}
    ships = sl.setdefault("ship", {})
    changes: list[tuple[str, float, float]] = []
    step = 0.03 if won else 0.02
    for row in seat_row.get("ships") or []:
        sid = str(row.get("ship_id", ""))
        if not sid:
            continue
        if content is not None and not is_shop_combat_hull(content.ships.get(sid)):
            continue
        out_v = float(row.get("dmg_out") or 0.0)
        in_v = float(row.get("dmg_in") or 0.0)
        old = float(ships.get(sid, 0.45))
        score = out_v - 0.35 * in_v
        delta = step if score > 0 else -step
        if out_v <= 0.0 and in_v > 0:
            delta = -step
        new = round(max(0.05, min(0.99, old + delta)), 4)
        ships[sid] = new
        if abs(new - old) > 1e-9:
            changes.append((sid, old, new))
    st = genome.setdefault("stance", {})
    if won:
        st["offense"] = round(min(0.45, float(st.get("offense", 0.2)) + 0.01), 4)
    else:
        st["logistics"] = round(min(0.45, float(st.get("logistics", 0.2)) + 0.01), 4)
    return changes


def main() -> None:
    cfg = load_config()
    content = Content(cfg=cfg)
    prior = load_bootstrap(content=content)
    rng = random.Random(1)
    kernel = BattleKernel(content)
    ga = derive_seat_genome(prior, rng, content)
    gb = derive_seat_genome(prior, rng, content)
    from eveac_ai.titan_draft import draft_two_rounds

    draft = draft_two_rounds([ga, gb], rng)
    titan_a, titan_b = draft["round2"][0], draft["round2"][1]
    fleet_a = pick_fleet(content, ga, titan_a, n=4)
    fleet_b = pick_fleet(content, gb, titan_b, n=4)
    slice_a = ((ga.get("titan_slices") or {}).get(titan_a) or {}).get("ship") or {}
    slice_b = ((gb.get("titan_slices") or {}).get(titan_b) or {}).get("ship") or {}
    before_a = {sid: float(slice_a.get(sid, 0.0)) for sid in fleet_a}
    before_b = {sid: float(slice_b.get(sid, 0.0)) for sid in fleet_b}
    stance_a0 = dict(ga.get("stance") or {})
    stance_b0 = dict(gb.get("stance") or {})

    pack = kernel.fight(
        fleet_a=fleet_a,
        fleet_b=fleet_b,
        seed=42,
        match_id="one-table",
        round_i=0,
        seat_a=0,
        seat_b=1,
    )
    row_a, row_b = pack["seats"][0], pack["seats"][1]
    ch_a = apply_telemetry_delta(ga, titan_a, row_a, bool(row_a["won"]), content)
    ch_b = apply_telemetry_delta(gb, titan_b, row_b, bool(row_b["won"]), content)

    out_dir = ROOT / "samples" / "one_table"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "telemetry.json").write_text(json.dumps(pack, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines: list[str] = []
    lines.append("=== 一桌试打 ===")
    lines.append(f"backend={pack.get('backend')} seed=42")
    lines.append(f"A 泰坦={titan_a} 上场={[ _names(content, s) for s in fleet_a ]}")
    lines.append(f"B 泰坦={titan_b} 上场={[ _names(content, s) for s in fleet_b ]}")
    lines.append(f"胜负: A won={row_a['won']} titan_hp={row_a['titan_hp']} | B won={row_b['won']} titan_hp={row_b['titan_hp']}")
    lines.append("")
    lines.append("--- 拟真明细（本席舰）---")
    for label, row in (("A", row_a), ("B", row_b)):
        lines.append(f"[{label}]")
        for sh in row.get("ships") or []:
            lines.append(
                f"  {_names(content, str(sh['ship_id']))} out={sh['dmg_out']:.1f} in={sh['dmg_in']:.1f} "
                f"cap={sh['cap_used']:.1f} lock_s={sh['lock_s']:.1f} survived={sh['survived']}"
            )
        kills = row.get("kill_calendar") or []
        lines.append(f"  击毁日历 n={len(kills)} {kills[:8]}")
    lines.append("")
    lines.append("--- 参数变化（吃拟真：输出高+ / 白挨打且无输出-；胜方 offense+ / 负方 logistics+）---")
    lines.append(f"A stance {stance_a0} -> {ga.get('stance')}")
    lines.append(f"B stance {stance_b0} -> {gb.get('stance')}")
    for label, titan, before, ch, genome in (
        ("A", titan_a, before_a, ch_a, ga),
        ("B", titan_b, before_b, ch_b, gb),
    ):
        lines.append(f"[{label} {titan} 切片 上场舰]")
        if not ch:
            lines.append("  (无变化)")
        for sid, old, new in ch:
            sign = "+" if new > old else ""
            lines.append(f"  {_names(content, sid)}  {old:.4f} -> {new:.4f} ({sign}{new-old:.4f})")
    text = "\n".join(lines) + "\n"
    log_path = out_dir / "log.txt"
    log_path.write_text(text, encoding="utf-8")
    (out_dir / "genome_a_after.json").write_text(json.dumps(ga, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "genome_b_after.json").write_text(json.dumps(gb, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(text)
    print(f"wrote {log_path}")


if __name__ == "__main__":
    main()
