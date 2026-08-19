"""One-off: C-FSM flagship jump rates from session/checkpoint/diag + simulation."""
from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from eveac_ai.content import Content, load_config
from eveac_ai.economy import grant_exp
from eveac_ai.match20 import fight_units
from eveac_ai.prepare import is_cyno_flagship, is_covert_cyno
from eveac_ai.scripted_fsm import prepare_miner_flag
from eveac_ai.seat_prep import new_seat_board


def analyze_seat_pieces(content: Content, pieces: list) -> dict:
    has_h = any(
        is_cyno_flagship(content.ships.get(str(p.get("ship_id")))) and p.get("slot") == "hangar"
        for p in pieces
    )
    has_c = any(
        is_covert_cyno(content.ships.get(str(p.get("ship_id")))) and p.get("slot") == "field"
        for p in pieces
    )
    fu = fight_units(pieces, content.ships)
    has_jump = any(u.get("cyno_hold") for u in fu)
    return {"hangar_flag": has_h, "field_cyno": has_c, "jump_fight": has_jump}


def main() -> None:
    content = Content(cfg=load_config())

    sess = json.loads((ROOT / "samples/session.json").read_text(encoding="utf-8"))
    fsm = sess.get("fsm_lifetime") or sess.get("fsm") or {}
    print("=== session.json fsm (latest batch) ===")
    print(json.dumps(fsm, ensure_ascii=False))
    fh, fj = int(fsm.get("flag_hangar") or 0), int(fsm.get("flag_jump") or 0)
    if fh:
        print(f"  prepare turns jump|hangar: {fj}/{fh} = {fj/fh:.1%}  (given hangar has flagship, field has cyno)")

    pat = re.compile(
        r"deploy gen=(\d+) L=(\d+) r=(\d+) seat=(\d+) scripted=miner_flag field=(\d+) hangar=(\d+)"
    )
    diag_lines = (ROOT / "samples/logs/diag.log").read_text(encoding="utf-8", errors="replace").splitlines()
    rows = [m.groups() for l in diag_lines if "scripted=miner_flag" in l for m in [pat.search(l)] if m]
    print(f"\n=== diag scripted=miner_flag deploy lines: {len(rows)} ===")
    by_gen: dict[int, int] = defaultdict(int)
    rounds: list[int] = []
    for g, _L, r, _sid, _f, _h in rows:
        by_gen[int(g)] += 1
        rounds.append(int(r))
    print("by gen:", dict(sorted(by_gen.items())))
    if rounds:
        print(f"round range r={min(rounds)}..{max(rounds)} (deploy lines only exist while seat alive)")

    ck_path = ROOT / "samples/match_checkpoint.json"
    if ck_path.is_file():
        ck = json.loads(ck_path.read_text(encoding="utf-8"))
        fsm_seats = []
        for lg in ck.get("leagues") or []:
            for s in lg.get("seats") or []:
                if s.get("scripted") == "miner_flag":
                    fsm_seats.append((lg.get("league_i"), s))
        print(f"\n=== match_checkpoint FSM seats at save: {len(fsm_seats)} ===")
        stats = defaultdict(int)
        levels: list[int] = []
        for _li, s in fsm_seats:
            board = s.get("board") or {}
            lv = int(board.get("level") or 1)
            levels.append(lv)
            pieces = board.get("pieces") or []
            row = analyze_seat_pieces(content, pieces)
            stats["n"] += 1
            if lv >= 15:
                stats["lv15"] += 1
            if row["hangar_flag"]:
                stats["hangar_flag"] += 1
            if row["field_cyno"]:
                stats["field_cyno"] += 1
            if row["jump_fight"]:
                stats["jump_fight"] += 1
        n = stats["n"] or 1
        print(f"  lv>=15: {stats['lv15']}/{n} ({stats['lv15']/n:.1%})")
        print(f"  hangar flagship: {stats['hangar_flag']}/{n} ({stats['hangar_flag']/n:.1%})")
        print(f"  field cyno: {stats['field_cyno']}/{n} ({stats['field_cyno']/n:.1%})")
        print(f"  fight_units cyno_hold: {stats['jump_fight']}/{n} ({stats['jump_fight']/n:.1%})")
        if stats["hangar_flag"]:
            print(f"  P(jump|hangar) end-state: {stats['jump_fight']}/{stats['hangar_flag']} ({stats['jump_fight']/stats['hangar_flag']:.1%})")
        if levels:
            print(f"  levels mean={sum(levels)/len(levels):.1f} min={min(levels)} max={max(levels)}")

    # pop line from diag
    pop_pat = re.compile(r"pop gen=(\d+).*?fsm=\{([^}]+)\}")
    pop_hits = [pop_pat.search(l) for l in diag_lines if "pop gen=" in l and "fsm=" in l]
    pop_hits = [h for h in pop_hits if h]
    if pop_hits:
        print("\n=== diag pop fsm aggregates ===")
        for h in pop_hits[-3:]:
            g = h.group(1)
            blob = "{" + h.group(2) + "}"
            blob = blob.replace("'", '"')
            try:
                st = json.loads(blob)
            except json.JSONDecodeError:
                st = {}
            fh2, fj2 = int(st.get("flag_hangar") or 0), int(st.get("flag_jump") or 0)
            wr = st.get("pvp_w"), st.get("pvp_l"), st.get("pvp_d")
            line = f"gen={g} seats={st.get('seats')} PVP={wr[0]}-{wr[1]}-{wr[2]}"
            if fh2:
                line += f" jump/hangar_prepare={fj2}/{fh2}={fj2/fh2:.1%}"
            print(" ", line)

    print("\n=== simulate 45 rounds (isolated FSM, +5 gold/rd income) ===")
    for seed in (1, 2, 3):
        rng = random.Random(seed)
        board = new_seat_board()
        prep_n = hang_prep = jump_prep = 0
        first = {"lv15": None, "hang": None, "jump": None}
        for rnd in range(45):
            board["gold"] = int(board.get("gold") or 0) + 5
            if int(board.get("level") or 1) < 15:
                grant_exp(content.economy, board, 4)
            prepare_miner_flag(content, {}, "amarr", board, rng, content.board)
            prep_n += 1
            lv = int(board.get("level") or 1)
            row = analyze_seat_pieces(content, board["pieces"])
            if lv >= 15 and first["lv15"] is None:
                first["lv15"] = rnd + 1
            if row["hangar_flag"]:
                hang_prep += 1
                if first["hang"] is None:
                    first["hang"] = rnd + 1
            if row["jump_fight"]:
                jump_prep += 1
                if first["jump"] is None:
                    first["jump"] = rnd + 1
        print(
            f"  seed={seed}: R_lv15={first['lv15']} R_hangar={first['hang']} R_jump={first['jump']} | "
            f"P(hangar|prep)={hang_prep/prep_n:.1%} P(jump|prep)={jump_prep/prep_n:.1%} "
            f"P(jump|hangar)={(jump_prep/hang_prep if hang_prep else 0):.1%} end_lv={board['level']}"
        )


if __name__ == "__main__":
    main()
