"""Deeper C-FSM flagship rate: per-prepare from diag + realistic sim."""
from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from eveac_ai.content import Content, load_config
from eveac_ai.economy import grant_exp, round_income_pre
from eveac_ai.match20 import fight_units
from eveac_ai.prepare import is_cyno_flagship, is_covert_cyno
from eveac_ai.scripted_fsm import prepare_miner_flag
from eveac_ai.seat_prep import new_seat_board


def seat_metrics(content: Content, pieces: list) -> tuple[bool, bool, bool]:
    hang = any(
        is_cyno_flagship(content.ships.get(str(p.get("ship_id")))) and p.get("slot") == "hangar"
        for p in pieces
    )
    cyno = any(
        is_covert_cyno(content.ships.get(str(p.get("ship_id")))) and p.get("slot") == "field"
        for p in pieces
    )
    jump = any(u.get("cyno_hold") for u in fight_units(pieces, content.ships))
    return hang, cyno, jump


def parse_diag_gen(gen: int) -> dict:
    pat = re.compile(
        rf"deploy gen={gen} L=(\d+) r=(\d+) seat=(\d+) scripted=miner_flag field=(\d+) hangar=(\d+)"
    )
    lines = (ROOT / "samples/logs/diag.log").read_text(encoding="utf-8", errors="replace").splitlines()
    rows = []
    for ln in lines:
        m = pat.search(ln)
        if m:
            rows.append(tuple(int(x) for x in m.groups()))
    return {"rows": rows, "n_prepare": len(rows)}


def sim_realistic(content: Content, *, seeds: range, rounds: int = 50, win_rate: float = 0.45) -> None:
    econ = content.economy
    print(f"\n=== realistic sim ({rounds} rds, win_rate={win_rate}) ===")
    for seed in seeds:
        rng = random.Random(seed)
        board = new_seat_board()
        board["win_streak"] = 0
        board["loss_streak"] = 0
        prep = hang_n = jump_n = cyno_n = lv15_n = 0
        first = {}
        for rnd in range(rounds):
            won = rng.random() < win_rate
            pre = round_income_pre(
                econ,
                gold_ref=int(board["gold"]),
                round_i=rnd,
                won=won,
                win_streak=int(board.get("win_streak") or 0),
                mining_g=0,
            )
            board["gold"] = int(board["gold"]) + int(pre["income"])
            if won:
                board["win_streak"] = int(board.get("win_streak") or 0) + 1
                board["loss_streak"] = 0
            else:
                board["loss_streak"] = int(board.get("loss_streak") or 0) + 1
                board["win_streak"] = 0
            grant_exp(econ, board, int(econ.get("base_exp_income") or 4))
            prepare_miner_flag(content, {}, "amarr", board, rng, content.board)
            prep += 1
            lv = int(board.get("level") or 1)
            hang, cyno, jump = seat_metrics(content, board["pieces"])
            if lv >= 15:
                lv15_n += 1
                first.setdefault("lv15", rnd + 1)
            if hang:
                hang_n += 1
                first.setdefault("hang", rnd + 1)
            if cyno:
                cyno_n += 1
            if jump:
                jump_n += 1
                first.setdefault("jump", rnd + 1)
        print(
            f"  seed={seed}: first R15={first.get('lv15')} Rh={first.get('hang')} Rj={first.get('jump')} "
            f"end_lv={board['level']} gold={board['gold']} | "
            f"P(lv15|prep)={lv15_n/prep:.1%} P(hang|prep)={hang_n/prep:.1%} "
            f"P(cyno|prep)={cyno_n/prep:.1%} P(jump|prep)={jump_n/prep:.1%} "
            f"P(jump|hang)={(jump_n/hang_n if hang_n else 0):.1%}"
        )
        flags = [p for p in board["pieces"] if is_cyno_flagship(content.ships.get(str(p["ship_id"])))]
        cynos = [p for p in board["pieces"] if is_covert_cyno(content.ships.get(str(p["ship_id"])))]
        if flags or cynos:
            print(f"    end pieces flags={[(p['ship_id'], p['slot']) for p in flags]} cynos={[(p['ship_id'], p['slot']) for p in cynos]}")


def main() -> None:
    content = Content(cfg=load_config())

    for gen in (68, 69):
        d = parse_diag_gen(gen)
        if not d["rows"]:
            continue
        rounds = [r for _L, r, _s, _f, _h in d["rows"]]
        print(f"\n=== gen={gen} miner_flag prepares logged: {d['n_prepare']} ===")
        print(f"  round span {min(rounds)}..{max(rounds)}  tables touched: {len({L for L, *_ in d['rows']})}")

    sess = json.loads((ROOT / "samples/session.json").read_text(encoding="utf-8"))
    fsm = sess.get("fsm_lifetime") or {}
    fh, fj = int(fsm.get("flag_hangar") or 0), int(fsm.get("flag_jump") or 0)
    n_prep68 = parse_diag_gen(68)["n_prepare"]
    print("\n=== gen68 rate estimates ===")
    print(f"  fsm flag_hangar prepare-count (farm stat): {fh}")
    print(f"  miner_flag deploy lines (proxy all prepares): {n_prep68}")
    if n_prep68:
        print(f"  P(hangar flagship | prepare) ≈ {fh/n_prep68:.1%}  (upper bound if all lines are unique prepares)")
    if fh:
        print(f"  P(jump fight | hangar flagship prepare) = {fj/fh:.1%}")

    # per-table: max round reached
    by_league: dict[int, int] = defaultdict(int)
    pat = re.compile(r"deploy gen=68 L=(\d+) r=(\d+) seat=\d+ scripted=miner_flag")
    for ln in (ROOT / "samples/logs/diag.log").read_text(encoding="utf-8", errors="replace").splitlines():
        m = pat.search(ln)
        if m:
            L, r = int(m.group(1)), int(m.group(2))
            by_league[L] = max(by_league[L], r)
    survived = [r for r in by_league.values()]
    if survived:
        print(f"  FSM seat survival (max round): mean={sum(survived)/len(survived):.1f} "
              f"median={sorted(survived)[len(survived)//2]} n_tables={len(survived)}")

    sim_realistic(content, seeds=range(5), win_rate=0.45)
    sim_realistic(content, seeds=range(5, 10), win_rate=0.55)

    ck_path = ROOT / "samples/match_checkpoint.json"
    if ck_path.is_file():
        ck = json.loads(ck_path.read_text(encoding="utf-8"))
        print("\n=== checkpoint FSM seat detail ===")
        for lg in ck.get("leagues") or []:
            for s in lg.get("seats") or []:
                if s.get("scripted") != "miner_flag":
                    continue
                b = s.get("board") or {}
                pcs = b.get("pieces") or []
                flags = [(p["ship_id"], p["slot"]) for p in pcs if is_cyno_flagship(content.ships.get(str(p["ship_id"])))]
                cynos = [(p["ship_id"], p["slot"]) for p in pcs if is_covert_cyno(content.ships.get(str(p["ship_id"])))]
                rorqs = [(p["ship_id"], p["slot"]) for p in pcs if str(p["ship_id"]) == "138"]
                fu = fight_units(pcs, content.ships)
                j = [u["ship_id"] for u in fu if u.get("cyno_hold")]
                fn = sum(1 for p in pcs if p.get("slot") == "field")
                print(
                    f"  L{lg.get('league_i')} seat={s.get('seat_id')} lv={b.get('level')} gold={b.get('gold')} "
                    f"w={s.get('wins')} hp={float(s.get('titan_hp') or 0):.0f} field={fn}"
                )
                print(f"    flags={flags} cynos={cynos} rorqual={rorqs} jump_fight={j}")


if __name__ == "__main__":
    main()
