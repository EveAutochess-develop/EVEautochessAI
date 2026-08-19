"""Re-sim farm snapshots on CPU kernel and optional Godot CombatResolver."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from eveac_ai.content import Content, load_config
from eveac_ai.kernel import BattleKernel

ROOT = Path(__file__).resolve().parents[1]
SNAP = ROOT / "samples" / "compare_snapshots"
GODOT_PROJECT = Path("H:/game_dev/eveautochess-dev/godot_project")


def _winner_from_pack(pack: dict) -> str:
    a, b = pack["seats"][0], pack["seats"][1]
    if a.get("won") and not b.get("won"):
        return "a"
    if b.get("won") and not a.get("won"):
        return "b"
    return "draw"


def _winner_from_ships(ships_a: list, ships_b: list) -> str:
    def live(ships: list) -> bool:
        return any((not s.get("is_unmanned")) and s.get("survived") for s in (ships or []))

    a_live, b_live = live(ships_a), live(ships_b)
    if a_live and not b_live:
        return "a"
    if b_live and not a_live:
        return "b"
    return "draw"


def _surv(ships: list) -> tuple[int, int]:
    manned = [s for s in ships if not s.get("is_unmanned")]
    live = [s for s in manned if s.get("survived")]
    return len(live), len(manned)


def run_cpu(job: dict) -> dict:
    k = BattleKernel(Content(cfg=load_config()))
    return k.fight(**{x: job[x] for x in job if x in (
        "fleet_a", "fleet_b", "pos_a", "pos_b", "titan_a", "titan_b", "seed", "match_id", "round_i", "seat_a", "seat_b"
    )})


def find_godot() -> str | None:
    candidates = [
        Path(r"H:\game_dev\eveautochess-dev\tools\godot\Godot_v4.7.1-stable_win64.exe"),
        Path(r"C:\Program Files\Godot\Godot_v4.exe"),
        Path(r"H:\Godot\Godot_v4.exe"),
    ]
    for p in candidates:
        if p.is_file():
            return str(p)
    try:
        r = subprocess.run(["where", "godot"], capture_output=True, text=True, timeout=10)
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip().splitlines()[0]
    except OSError:
        pass
    for p in Path(r"H:\game_dev").glob("**/Godot*.exe"):
        if "console" in p.name.lower() or p.name.lower().startswith("godot"):
            return str(p)
    return None


def run_godot(snap_path: Path, out_path: Path, godot: str) -> dict | None:
    script = GODOT_PROJECT / "tools" / "farm_snapshot_combat.gd"
    if not script.is_file():
        return None
    job_file = GODOT_PROJECT / "tools" / "_farm_snapshot_job.json"
    job_file.write_text(
        json.dumps(
            {"snapshot": snap_path.resolve().as_posix(), "out": out_path.resolve().as_posix()},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    cmd = [
        godot,
        "--headless",
        "--path",
        str(GODOT_PROJECT),
        "--scene",
        "res://tools/farm_snapshot_combat.tscn",
        "--",
        f"--snapshot={snap_path.resolve().as_posix()}",
        f"--out={out_path.resolve().as_posix()}",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=960)
    if not out_path.is_file():
        return {"ok": False, "stderr": (r.stderr or r.stdout)[-2000:]}
    return json.loads(out_path.read_text(encoding="utf-8"))


def main() -> None:
    files = sorted(p for p in SNAP.glob("*.json") if not p.name.startswith("_"))
    if not files:
        print("no snapshots in", SNAP)
        sys.exit(1)
    godot = find_godot()
    print(f"snapshots={len(files)} godot={godot or 'none'}")
    agree_cpu_godot = 0
    agree_rec_cpu = 0
    n_godot = 0
    n = min(30, len(files))
    rows = []
    for p in files[:n]:
        blob = json.loads(p.read_text(encoding="utf-8"))
        farm = blob["farm"]
        job = blob["job"]
        rec_w = _winner_from_ships(farm.get("ships_a") or [], farm.get("ships_b") or [])
        if farm.get("draw"):
            rec_w = "draw"
        elif farm.get("a_won") is not None and farm.get("b_won") is not None:
            if farm.get("a_won") and not farm.get("b_won"):
                rec_w = "a"
            elif farm.get("b_won") and not farm.get("a_won"):
                rec_w = "b"
            elif not farm.get("a_won") and not farm.get("b_won"):
                rec_w = "draw"
        cpu = run_cpu(job)
        cpu_w = _winner_from_pack(cpu)
        sa, na = _surv(cpu["seats"][0].get("ships") or [])
        sb, nb = _surv(cpu["seats"][1].get("ships") or [])
        agree_rec_cpu += int(rec_w == cpu_w)
        g_w = "-"
        g_note = ""
        if godot:
            gout = SNAP / "_godot" / (p.stem + ".godot.json")
            gout.parent.mkdir(parents=True, exist_ok=True)
            if gout.is_file():
                gout.unlink()
            gr = run_godot(p, gout, godot)
            if gr and gr.get("ok"):
                n_godot += 1
                g_w = str(gr.get("winner") or "")
                g_note = f"sim_s={gr.get('sim_s')} reason={gr.get('reason')} live_a={gr.get('live_a')} live_b={gr.get('live_b')}"
                if g_w == cpu_w:
                    agree_cpu_godot += 1
            else:
                g_note = str((gr or {}).get("stderr") or "godot fail")[:180]
        line = (
            f"{p.name} rec={rec_w}({farm.get('backend')}) "
            f"cpu={cpu_w} live={sa}/{na} vs {sb}/{nb} "
            f"godot={g_w} {g_note}"
        )
        print(line, flush=True)
        rows.append(line)
    print(f"rec_vs_cpu {agree_rec_cpu}/{n}")
    if n_godot:
        print(f"cpu_vs_godot {agree_cpu_godot}/{n_godot} ({100.0 * agree_cpu_godot / n_godot:.0f}%)")
    (SNAP / "compare_report.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
