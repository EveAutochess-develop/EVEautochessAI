"""Nullsec/lowsec self-play farm (multi-round until ranking)."""

from __future__ import annotations

import json
import random
import signal
import sys
import time
from pathlib import Path

from eveac_ai.capped_log import CappedLog
from eveac_ai.economy import grant_exp, loss_comp, mining_gold_from_survivors, round_income_pre
from eveac_ai.prepare import fleet_ids_from_pieces, is_cyno_flagship, is_covert_cyno
from eveac_ai.ranking import format_ranking_delta, format_ranking_table, item_key, quantize_ships, ranking_index, spend_catalog
from eveac_ai.board_view import ship_label
from eveac_ai.content import Content, load_config, remaining_pvp_losses, titan_max_hp, titan_pvp_loss
from eveac_ai.device import resolve_device
from eveac_ai.elite_pool import draw_frozen_genomes, save_generation as save_elite_generation
from eveac_ai.farm_lock import acquire_farm_lock, consume_stop_file, release_farm_lock
from eveac_ai.gpu_kernel import GpuBattleKernel
from eveac_ai.kernel import BattleKernel
from eveac_ai.nets.pack import FourNetPack, pair_advantage
from eveac_ai.telemetry_credit import (
    apply_genome_delta,
    cyno_key_delta,
    ema_blend,
    eval_valence_delta,
    first_kill_s,
    merge_ship_equip_credit,
    path_value,
    source_weight,
    unit_credit,
)
from eveac_ai.orchestrator import blend_genomes
from eveac_ai.priors import derive_seat_genome, load_bootstrap, scrub_genome_ships
from eveac_ai.replay import ReplayWriter
from eveac_ai.pve import is_pvp_round, lock_creeps, pve_success, roll_pve_task, salvage_freighter_unit
from eveac_ai.seat_prep import _pop_cap, new_seat_board, prepare_turn
from eveac_ai.sim_stats import add_summaries, format_gen_stats, format_round_stats, summary_from_timing
from eveac_ai.scripted_fsm import (
    add_fsm_stats,
    empty_fsm_stats,
    format_fsm_feedback,
    league_fsm_stats,
    pick_scripted_seat_ids,
    SCRIPTED_KIND,
)
from eveac_ai.nets.memory import _cost, hydrate_memory, new_memory, note_fight, round_summary
from eveac_ai.state_bank import ingest_checkpoint
from eveac_ai.stdio_utf8 import configure_utf8_stdio
from eveac_ai.titan_draft import draft_two_rounds, draft_with_net

ROOT = Path(__file__).resolve().parents[1]


# region agent log
def _agent_dbg(hypothesis_id: str, location: str, message: str, data: dict) -> None:
    try:
        rec = {
            "sessionId": "519d6e",
            "timestamp": int(time.time() * 1000),
            "hypothesisId": hypothesis_id,
            "runId": "post-fix",
            "location": location,
            "message": message,
            "data": data,
        }
        with Path(r"H:\debug-519d6e.log").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass


# endregion


def titan_loss_of(content: Content) -> float:
    return titan_pvp_loss(content.titan_pvp)


def _pvp_loss_mul(cfg: dict) -> float:
    mode = str(cfg.get("_security_mode") or cfg.get("security_mode") or "nullsec")
    if mode == "lowsec":
        return float(cfg.get("lowsec_pvp_loss_mul") or 0.25)
    return 1.0


def _scaled_titan_loss(content: Content, cfg: dict) -> float:
    raw = titan_loss_of(content)
    mul = _pvp_loss_mul(cfg)
    return float(max(1, int(round(raw * mul))))


def _job_unit_count(job: dict) -> int:
    pa = job.get("pos_a") or []
    pb = job.get("pos_b") or []
    if pa or pb:
        return len(pa) + len(pb)
    return len(job.get("fleet_a") or []) + len(job.get("fleet_b") or [])


def route_fight_batch(
    jobs: list[dict],
    *,
    kernel: BattleKernel | GpuBattleKernel,
    use_gpu: bool,
    cfg: dict,
) -> tuple[list[dict], str]:
    """Auto CPU vs GPU: small B or B*S → CPU event kernel; dense mass → GPU batch."""
    if not jobs:
        # region agent log
        _agent_dbg(
            "D",
            "match20.py:route_fight_batch:empty",
            "empty jobs skip fight",
            {
                "route": "none",
                "B": 0,
                "wrapper_sim": float(getattr(kernel, "last_sim_s", 0.0) or 0.0),
                "cpu_sim": float(getattr(getattr(kernel, "cpu", kernel), "last_sim_s", 0.0) or 0.0),
            },
        )
        # endregion
        return [], "none"
    B = len(jobs)
    S = max((_job_unit_count(j) for j in jobs), default=1)
    min_jobs = int(cfg.get("gpu_min_jobs") or 24)
    min_bs = int(cfg.get("gpu_min_BS") or 400)
    want = bool(use_gpu) and B >= min_jobs and (B * max(S, 1)) >= min_bs
    if want and isinstance(kernel, GpuBattleKernel):
        t0 = time.perf_counter()
        packs = kernel.fight_batch(jobs, slots=int(cfg.get("sim_parallel") or cfg.get("cuda_parallel_matches") or 0) or None)
        wall_s = time.perf_counter() - t0
        # region agent log
        timing = dict(getattr(kernel, "last_timing", None) or {})
        timing.update(
            {
                "route": "gpu",
                "job_S": S,
                "want": want,
                "use_gpu": bool(use_gpu),
                "n_packs": len(packs),
                "n_jobs": len(jobs),
                "wrapper_sim": float(getattr(kernel, "last_sim_s", 0.0) or 0.0),
                "cpu_sim": float(getattr(kernel.cpu, "last_sim_s", 0.0) or 0.0),
                "wall_s": round(wall_s, 3),
                "x_batch": round(float(getattr(kernel, "last_sim_s", 0.0) or 0.0) / max(wall_s, 0.001), 2),
                "arch_sparse_on": bool(timing.get("sparse")),
                "arch_sparse_expected": int(timing.get("S_kernel") or S) <= 24,
                "arch_route": "gpu",
            }
        )
        _agent_dbg("F", "match20.py:route_fight_batch:gpu", "gpu batch timing vs architecture", timing)
        # endregion
        return packs, "gpu"
    cpu = kernel.cpu if isinstance(kernel, GpuBattleKernel) else kernel
    packs = []
    sim_times: list[float] = []
    finishes: list[tuple[float, float]] = []
    occupy_s = 0.0
    t0 = time.perf_counter()
    for j in jobs:
        t1 = time.perf_counter()
        packs.append(cpu.fight(**j))
        sim = float(getattr(cpu, "last_sim_s", 0.0) or 0.0)
        t2 = time.perf_counter()
        sim_times.append(sim)
        finishes.append((t2 - t0, sim))
        occupy_s += t2 - t1
    wall_s = time.perf_counter() - t0
    sim_max = max(sim_times) if sim_times else 0.0
    mean_sim = (sum(sim_times) / len(sim_times)) if sim_times else 0.0
    kernel.last_sim_s = float(sim_max)
    kernel.last_timing = {
        "end_mean": round(mean_sim, 3),
        "end_min": round(min(sim_times), 3) if sim_times else None,
        "end_max": round(sim_max, 3),
        "n_cap": sum(1 for x in sim_times if x >= 899.0),
        "B": B,
        "n_jobs": B,
        "slots": 1,
        "slotted": False,
        "n_refill": 0,
        "S_kernel": S,
        "sparse": False,
        "sparse_gate_S24": S <= 24,
        "wall_s": round(wall_s, 3),
        "occupy_s": round(occupy_s, 3),
        "finishes": [(round(w, 3), round(s, 3)) for w, s in finishes],
    }
    # region agent log
    zero_n = sum(1 for t in sim_times if t < 0.5)
    sub1_n = sum(1 for t in sim_times if t < 1.25)
    _agent_dbg(
        "F",
        "match20.py:route_fight_batch:cpu",
        "cpu wave timing vs wrapper",
        {
            "route": "cpu",
            "B": B,
            "S": S,
            "want": want,
            "use_gpu": bool(use_gpu),
            "wrapper_is_gpu": isinstance(kernel, GpuBattleKernel),
            "wrapper_sim": float(getattr(kernel, "last_sim_s", 0.0) or 0.0),
            "cpu_sim": float(getattr(cpu, "last_sim_s", 0.0) or 0.0),
            "fight_min": min(sim_times) if sim_times else None,
            "fight_max": sim_max if sim_times else None,
            "fight_mean": mean_sim if sim_times else None,
            "zero_lt_0_5": zero_n,
            "early_lt_min_s": sub1_n,
            "wall_s": round(wall_s, 3),
            "x_mean": round(mean_sim / max(wall_s, 0.001), 2) if sim_times else None,
            "arch_route_expected": "gpu" if want else "cpu",
            "synced_wrapper": True,
        },
    )
    # endregion
    return packs, "cpu"


def _should_write_checkpoint(cfg: dict, round_1based: int, *, force: bool = False) -> bool:
    if force:
        return True
    mode = str(cfg.get("_security_mode") or "nullsec")
    if mode == "lowsec":
        landmarks = {int(x) for x in (cfg.get("lowsec_landmarks") or [3, 6, 10])}
    else:
        landmarks = {int(x) for x in (cfg.get("nullsec_landmarks") or [4, 6, 8])}
    every = max(1, int(cfg.get("checkpoint_every_n") or 5))
    return round_1based in landmarks or (round_1based % every == 0)


def _elite_key(s: dict, titan_pvp: dict | None = None) -> tuple:
    lives = remaining_pvp_losses(float(s.get("titan_hp") or 0), titan_pvp)
    return (-int(s["alive"]), -int(s["wins"]), -lives, int(s.get("losses") or 0), str(s["seat_id"]))


def combat_units(content: Content, pieces: list[dict], row: dict) -> list[dict]:
    ships = row.get("ships") or []
    if ships:
        out: list[dict] = []
        for s in ships:
            hull = content.ships.get(str(s.get("ship_id"))) or {}
            out.append(
                {
                    "ship_id": str(s.get("ship_id")),
                    "star": int(s.get("star") or 1),
                    "survived": bool(s.get("survived", True)),
                    "is_unmanned": bool(s.get("is_unmanned", hull.get("is_unmanned"))),
                    "unmanned_kind": str(s.get("unmanned_kind") or hull.get("unmanned_kind") or ""),
                }
            )
        return out
    pos = fight_units(pieces, content.ships)
    out = []
    for p in pos:
        hull = content.ships.get(str(p["ship_id"])) or {}
        out.append(
            {
                "ship_id": str(p["ship_id"]),
                "star": int(p.get("star") or 1),
                "survived": True,
                "is_unmanned": bool(hull.get("is_unmanned")),
                "unmanned_kind": str(hull.get("unmanned_kind") or ""),
            }
        )
    return out


def kill_gold_of(content: Content, opp_units: list[dict]) -> int:
    n = sum(1 for u in opp_units if (not u.get("survived")) and (not u.get("is_unmanned")))
    return n * int(content.economy.get("kill_gold_per_ship") or 1)



def _names(content: Content, sid: str) -> str:
    return ship_label(content, sid)


def pair_alive(alive: list[int], rng: random.Random) -> tuple[list[tuple[int, int]], int | None]:
    order = list(alive)
    rng.shuffle(order)
    bye = None
    if len(order) % 2 == 1:
        bye = order.pop()
    pairs = [(order[i], order[i + 1]) for i in range(0, len(order), 2)]
    return pairs, bye


def fight_units(pieces: list[dict], ships: dict) -> list[dict]:
    field = [p for p in pieces if p.get("slot") == "field"]
    has_cyno = any(is_covert_cyno(ships.get(str(p["ship_id"]))) for p in field)
    units = list(field)
    if has_cyno:
        units.extend(
            p
            for p in pieces
            if p.get("slot") == "hangar" and is_cyno_flagship(ships.get(str(p["ship_id"])))
        )
    out = []
    for p in units:
        hull = ships.get(str(p["ship_id"])) or {}
        out.append(
            {
                "token": p.get("token"),
                "ship_id": str(p["ship_id"]),
                "x": int(p.get("x") or 0),
                "z": int(p.get("z") or 0),
                "star": int(p.get("star") or 1),
                "equips": list(p.get("equips") or []),
                "cyno_hold": bool(is_cyno_flagship(hull)),
            }
        )
    return out


class MatchLeague:
    """One 20-seat league. Collects CUDA jobs so many leagues share one fight_batch."""

    def __init__(
        self,
        *,
        content: Content,
        cfg: dict,
        infer,
        use_gpu: bool,
        base: dict,
        seed: int,
        gen_i: int,
        league_i: int,
        diag: CappedLog,
        replay: ReplayWriter,
        replay_log: CappedLog,
        nets: FourNetPack | None = None,
    ) -> None:
        self.nets = nets
        self.place_net = None
        self.content = content
        self.cfg = cfg
        self.seed = int(seed)
        self.gen_i = gen_i
        self.league_i = league_i
        self.diag = diag
        self.replay = replay
        self.replay_log = replay_log
        self.keep = int(cfg.get("keep_top", 3))
        self.security_mode = str(cfg.get("_security_mode") or cfg.get("security_mode") or "nullsec")
        self.train_source = self.security_mode
        self.force_pvp = self.security_mode == "lowsec"
        self._trace_paired = False
        self.stats: dict[str, float] = {
            "xp_buys": 0,
            "prepares": 0,
            "field_full": 0,
            "skipped_xp": 0,
            "cyno_field": 0,
            "flag_hangar": 0,
            "flag_jump": 0,
            "level_sum": 0,
            "level15": 0,
            "fsm_flag_jump": 0,
            "fsm_flag_hangar": 0,
        }
        if self.security_mode == "lowsec":
            self.n_seats = int(cfg.get("lowsec_seats") or 2)
            self.keep = max(1, min(self.keep, 1))
        else:
            self.n_seats = int(cfg.get("seats", 20))
        self.rng = random.Random(int(seed))
        genomes = [derive_seat_genome(base, self.rng, self.content) for _ in range(self.n_seats)]
        scripted_ids = pick_scripted_seat_ids(
            self.n_seats,
            float(cfg.get("scripted_fsm_frac") or 0.0),
            self.rng,
            security_mode=self.security_mode,
        )
        frac = float(cfg.get("elite_pool_frac") or 0.15)
        n_frozen = max(0, int(round(self.n_seats * frac)))
        frozen_ids: set[int] = set(scripted_ids)
        remaining = [i for i in range(self.n_seats) if i not in scripted_ids]
        if n_frozen > 0 and remaining:
            frozen_gs = draw_frozen_genomes(ROOT / "samples", n_frozen, self.rng)
            if frozen_gs:
                n_elite = min(n_frozen, len(remaining), len(frozen_gs))
                elite_ids = set(self.rng.sample(remaining, n_elite))
                frozen_ids |= elite_ids
                for i, sid in enumerate(sorted(elite_ids)):
                    genomes[sid] = scrub_genome_ships(dict(frozen_gs[i % len(frozen_gs)]), self.content)
        if nets is not None:
            draft = draft_with_net(nets, genomes, self.rng)
        else:
            draft = draft_two_rounds(genomes, self.rng)
        self.draft = draft
        self.seen: set[str] = set()
        self.seats: list[dict] = []
        for i in range(self.n_seats):
            titan = draft["round2"][i]
            genome = genomes[i]
            board = new_seat_board()
            self.seats.append(
                {
                    "seat_id": i,
                    "titan": titan,
                    "titan_round1": draft["round1"][i],
                    "genome": genome,
                    "fleet": [],
                    "board": board,
                    "titan_hp": titan_max_hp(self.content.titan_pvp),
                    "wins": 0,
                    "losses": 0,
                    "alive": True,
                    "gold": board["gold"],
                    "memory": new_memory(),
                    "titan_lps": (draft.get("titan_lps") or [[None, None]] * self.n_seats)[i],
                    "frozen": i in frozen_ids,
                    "scripted": SCRIPTED_KIND if i in scripted_ids else "",
                }
            )
        self.match_id = f"gen{gen_i}L{league_i}"
        self.done = False
        diag.write(
            f"diag gen={gen_i} L={league_i} start infer={infer.kind}:{infer.name} seed={seed} seats={self.n_seats} "
            f"fsm={len(scripted_ids)} census1={draft['census']} kept={draft['kept']}"
        )
        replay.match_open(match_id=self.match_id, n_seats=self.n_seats, draft=draft, seats=self.seats)

    def _medians(self) -> tuple[float, float]:
        alive = [s for s in self.seats if s.get("alive")]
        if not alive:
            return 1.0, 1.0
        lvs = sorted(int((s.get("board") or {}).get("level") or 1) for s in alive)
        pops = sorted(
            sum(1 for p in ((s.get("board") or {}).get("pieces") or []) if p.get("slot") == "field") for s in alive
        )
        return float(lvs[len(lvs) // 2]), float(pops[len(pops) // 2])

    def _path_for(self, sid: int) -> float:
        s = self.seats[sid]
        board = s.get("board") or {}
        pieces = board.get("pieces") or []
        med_lv, med_pop = self._medians()
        tp = self.content.titan_pvp
        return path_value(
            level=int(board.get("level") or 1),
            field_n=sum(1 for p in pieces if p.get("slot") == "field"),
            lives=remaining_pvp_losses(float(s.get("titan_hp") or 0), tp),
            max_lives=max(1, remaining_pvp_losses(titan_max_hp(tp), tp)),
            med_level=med_lv,
            med_pop=med_pop,
        )

    def _remember_trace(self, seat: dict, trace: dict | None, adv: float, sw: float) -> None:
        if self.nets is None or not trace or seat.get("frozen"):
            return
        pieces = (seat.get("board") or {}).get("pieces") or []
        tagged = dict(trace)
        tagged["fight_adv"] = float(adv)
        tagged["path_value"] = self._path_for(int(seat.get("seat_id") or 0))
        tagged["cyno_key"] = cyno_key_delta(self.content, pieces)
        tagged["source_weight"] = sw
        prev = seat.pop("prev_trace", None)
        if prev:
            prev_ops = self.nets._live_lp(prev.get("ops_lp"))
            if prev_ops is not None:
                back = dict(prev)
                back["path_value"] = tagged["path_value"]
                back["source_weight"] = float(prev.get("source_weight") or sw) * float(self.nets.collab.get("path_gamma") or 0.85)
                self.nets.remember(back, float(prev.get("fight_adv") or adv))
        self.nets.remember(tagged, adv)
        self._trace_paired = True
        seat["prev_trace"] = tagged

    def flush_dead_traces(self) -> None:
        """Fold dead seats' leftover graphs into this backward (weights not stepped yet)."""
        if self.nets is None:
            return
        for s in self.seats:
            if s.get("alive"):
                continue
            prev = s.pop("prev_trace", None)
            if prev:
                prev["path_value"] = self._path_for(int(s.get("seat_id") or 0))
                self.nets.remember(prev, float(prev.get("fight_adv") or 0.0))

    def flush_pending_traces(self) -> None:
        """Train leftover stashes at match end (no opt.step since they were created)."""
        if self.nets is None:
            return
        for s in self.seats:
            prev = s.pop("prev_trace", None)
            if not prev:
                continue
            prev["path_value"] = self._path_for(int(s.get("seat_id") or 0))
            self.nets.remember(prev, float(prev.get("fight_adv") or 0.0))

    def _prepare_one(self, sid: int, rnd: int, traces: dict, pieces: dict, gold_before: dict, *, round_kind: str) -> None:
        s = self.seats[sid]
        traces[sid] = prepare_turn(
            self.content, s["genome"], s["titan"], s["board"], self.rng, self.content.board,
            self.nets, seat_id=sid, rnd=rnd, titan_hp=float(s["titan_hp"]), seats=self.seats,
            round_kind=round_kind, security_mode=self.security_mode,
            scripted=str(s.get("scripted") or "") or None,
        )
        tr = traces[sid] or {}
        self.stats["prepares"] += 1
        self.stats["level_sum"] += int(s["board"].get("level") or 1)
        if int(s["board"].get("level") or 1) >= 15:
            self.stats["level15"] += 1
        if tr.get("bought_xp"):
            self.stats["xp_buys"] += 1
        if tr.get("field_full"):
            self.stats["field_full"] += 1
        if tr.get("skipped_xp_when_full"):
            self.stats["skipped_xp"] += 1
        pieces[sid] = s["board"]["pieces"]
        pcs = pieces[sid]
        if any(is_covert_cyno(self.content.ships.get(str(p["ship_id"]))) and p.get("slot") == "field" for p in pcs):
            self.stats["cyno_field"] += 1
        if any(is_cyno_flagship(self.content.ships.get(str(p["ship_id"]))) and p.get("slot") == "hangar" for p in pcs):
            self.stats["flag_hangar"] += 1
            if s.get("scripted"):
                self.stats["fsm_flag_hangar"] += 1
            if any(is_covert_cyno(self.content.ships.get(str(p["ship_id"]))) and p.get("slot") == "field" for p in pcs):
                self.stats["flag_jump"] += 1
                if s.get("scripted"):
                    self.stats["fsm_flag_jump"] += 1
        s["fleet"] = fleet_ids_from_pieces(pieces[sid], self.content.ships)
        s["gold"] = int(s["board"]["gold"])
        gold_before[sid] = int(s["gold"])
        if getattr(self, "train_source", "") != "capital_bank" and not s.get("scripted"):
            self.seen.update(str(x) for x in s["fleet"])
            for p in pieces[sid]:
                self.seen.add(item_key("ship", str(p["ship_id"])))
                for eq in p.get("equips") or []:
                    self.seen.add(item_key("equip", str(eq).split(":", 1)[0]))
        fa = sum(1 for p in pieces[sid] if p["slot"] == "field")
        ha = sum(1 for p in pieces[sid] if p["slot"] == "hangar")
        eqa = sum(len(p.get("equips") or []) for p in pieces[sid])
        self.diag.write(
            f"deploy gen={self.gen_i} L={self.league_i} r={rnd+1} seat={sid} "
            f"scripted={s.get('scripted') or '-'} field={fa} hangar={ha} fit={eqa} gold={s['gold']}"
        )

    def _pve_fight_job(self, sid: int, rnd: int) -> dict:
        s = self.seats[sid]
        pieces = s["board"]["pieces"]
        field_value = sum(
            int(float((self.content.ships.get(str(p["ship_id"])) or {}).get("cost") or 0) or 0)
            for p in pieces
            if p.get("slot") == "field"
        )
        creeps = lock_creeps(
            self.content,
            self.rng,
            gold=int(s["gold"]),
            level=int(s["board"].get("level") or 1),
            pop_limit=_pop_cap(s["board"]),
            field_value=field_value,
        )
        task = str(getattr(self, "_round_kind", "") or "")
        if task not in ("pve_eliminate", "pve_salvage"):
            task = roll_pve_task(self.rng, rnd + 1, self.seed)
        pos_a = fight_units(pieces, self.content.ships)
        freighter_id = ""
        if task == "pve_salvage":
            fu = salvage_freighter_unit(self.content, self.rng, str(s.get("titan") or ""))
            freighter_id = str(fu["ship_id"])
            pos_a = list(pos_a) + [fu]
        pos_b = [
            {
                "ship_id": str(c["ship_id"]),
                "x": int(c.get("x") or 0),
                "z": int(c.get("z") or 0),
                "star": 1,
                "equips": [],
            }
            for c in creeps
        ]
        fleet_a = [str(p["ship_id"]) for p in pos_a]
        fleet_b = [str(c["ship_id"]) for c in pos_b]
        return {
            "fleet_a": fleet_a,
            "fleet_b": fleet_b,
            "pos_a": pos_a,
            "pos_b": pos_b,
            "titan_a": s["titan"],
            "titan_b": "",
            "seed": self.seed + rnd * 1009 + sid * 31,
            "match_id": self.match_id,
            "round_i": rnd,
            "seat_a": sid,
            "seat_b": -1,
            "pve_task": task,
            "freighter_id": freighter_id,
        }

    def collect_jobs(self, rnd: int) -> tuple[list[dict], dict | None]:
        if self.done:
            return [], None
        alive = [s["seat_id"] for s in self.seats if s["alive"]]
        if len(alive) <= self.keep:
            self.replay_log.write(f"{self.match_id} 第 {rnd} 轮前剩余 {len(alive)} 席，结束。")
            self.done = True
            return [], None
        for sid in alive:
            board = self.seats[sid]["board"]
            self.seats[sid]["gold"] = int(board["gold"])
        jobs: list[dict] = []
        pieces: dict[int, list] = {}
        traces: dict[int, dict] = {}
        gold_before: dict[int, int] = {}
        pve_jobs: list[dict] = []
        pairs: list[tuple[int, int]] = []
        bye = None
        force_pvp = bool(getattr(self, "force_pvp", False)) or self.security_mode == "lowsec"
        if (not force_pvp) and (not is_pvp_round(rnd + 1)):
            kind = roll_pve_task(self.rng, rnd + 1, self.seed)
            self._round_kind = kind
            self.diag.write(f"pve gen={self.gen_i} L={self.league_i} round={rnd+1} n={len(alive)} task={kind}")
            for sid in alive:
                self._prepare_one(sid, rnd, traces, pieces, gold_before, round_kind=kind)
                job = self._pve_fight_job(sid, rnd)
                jobs.append(job)
                pve_jobs.append(job)
        else:
            self._round_kind = "pvp"
            pairs, bye = pair_alive(alive, self.rng)
            self.diag.write(f"pair gen={self.gen_i} L={self.league_i} round={rnd+1} n={len(pairs)} bye={bye}")
            for a, b in pairs:
                self._prepare_one(a, rnd, traces, pieces, gold_before, round_kind="pvp")
                self._prepare_one(b, rnd, traces, pieces, gold_before, round_kind="pvp")
                sa, sb = self.seats[a], self.seats[b]
                jobs.append(
                    {
                        "fleet_a": sa["fleet"],
                        "fleet_b": sb["fleet"],
                        "pos_a": fight_units(pieces[a], self.content.ships),
                        "pos_b": fight_units(pieces[b], self.content.ships),
                        "titan_a": sa["titan"],
                        "titan_b": sb["titan"],
                        "seed": self.seed + rnd * 1009 + a * 17 + b,
                        "match_id": self.match_id,
                        "round_i": rnd,
                        "seat_a": a,
                        "seat_b": b,
                    }
                )
            if bye is not None and not force_pvp:
                bye_kind = roll_pve_task(self.rng, rnd + 1, self.seed)
                self._round_kind = bye_kind
                self._prepare_one(bye, rnd, traces, pieces, gold_before, round_kind=bye_kind)
                job = self._pve_fight_job(bye, rnd)
                jobs.append(job)
                pve_jobs.append(job)
            elif bye is not None and force_pvp:
                # Lowsec: bye sits out combat but still prepares (economy tick via skip).
                self._prepare_one(bye, rnd, traces, pieces, gold_before, round_kind="pvp")
        return jobs, {
            "pairs": pairs,
            "pve_jobs": pve_jobs,
            "pieces": pieces,
            "gold_before": gold_before,
            "bye": bye,
            "rnd": rnd,
            "traces": traces,
            "jobs": jobs,
        }

    def apply_packs(self, ctx: dict, packs: list[dict]) -> None:
        self._trace_paired = False
        rnd = int(ctx["rnd"])
        pieces = ctx["pieces"]
        gold_before = ctx["gold_before"]
        n_pvp = len(ctx.get("pairs") or [])
        for (a, b), pack in zip(ctx["pairs"], packs[:n_pvp]):
            sa, sb = self.seats[a], self.seats[b]
            row_a, row_b = pack["seats"][0], pack["seats"][1]
            traces = ctx.get("traces") or {}
            a_won = bool(row_a.get("won"))
            b_won = bool(row_b.get("won"))
            draw = (not a_won) and (not b_won)
            if draw:
                # Godot draw: both take titan loss; no win/loss streak winner.
                for sid in (a, b):
                    self.seats[sid]["losses"] += 1
                    self.seats[sid]["board"]["loss_streak"] = int(self.seats[sid]["board"].get("loss_streak") or 0) + 1
                    self.seats[sid]["board"]["win_streak"] = 0
                winner = a  # unused for econ paths below
                loser = b
            else:
                winner = a if a_won else b
                loser = b if a_won else a
                self.seats[winner]["wins"] += 1
                self.seats[loser]["losses"] += 1
                self.seats[winner]["board"]["win_streak"] = int(self.seats[winner]["board"].get("win_streak") or 0) + 1
                self.seats[winner]["board"]["loss_streak"] = 0
                self.seats[loser]["board"]["loss_streak"] = int(self.seats[loser]["board"].get("loss_streak") or 0) + 1
                self.seats[loser]["board"]["win_streak"] = 0
            units_a = combat_units(self.content, pieces[a], row_a)
            units_b = combat_units(self.content, pieces[b], row_b)
            mine_a = mining_gold_from_survivors(self.content, units_a)
            mine_b = mining_gold_from_survivors(self.content, units_b)
            kill_a = kill_gold_of(self.content, units_b)
            kill_b = kill_gold_of(self.content, units_a)
            econ = self.content.economy
            tp = self.content.titan_pvp
            pre_a = round_income_pre(
                econ,
                gold_ref=int(gold_before[a]) + kill_a,
                round_i=rnd,
                won=a_won,
                win_streak=int(self.seats[a]["board"]["win_streak"]),
                mining_g=mine_a,
            )
            pre_b = round_income_pre(
                econ,
                gold_ref=int(gold_before[b]) + kill_b,
                round_i=rnd,
                won=b_won,
                win_streak=int(self.seats[b]["board"]["win_streak"]),
                mining_g=mine_b,
            )
            if draw:
                win_field = 0
                win_pre = pre_a
                lose_pre = pre_b
                lc_a = loss_comp(
                    econ,
                    loss_streak=int(self.seats[a]["board"]["loss_streak"]),
                    winner_income=int(pre_b["income"]),
                    winner_field_cost=sum(
                        int((self.content.ships.get(str(p["ship_id"])) or {}).get("cost") or 0)
                        for p in pieces[b]
                        if p.get("slot") == "field"
                    ),
                    titan_pvp=tp,
                    loser_core=int(pre_a["base"]) + int(pre_a["interest"]) + int(pre_a["mining"]),
                )
                lc_b = loss_comp(
                    econ,
                    loss_streak=int(self.seats[b]["board"]["loss_streak"]),
                    winner_income=int(pre_a["income"]),
                    winner_field_cost=sum(
                        int((self.content.ships.get(str(p["ship_id"])) or {}).get("cost") or 0)
                        for p in pieces[a]
                        if p.get("slot") == "field"
                    ),
                    titan_pvp=tp,
                    loser_core=int(pre_b["base"]) + int(pre_b["interest"]) + int(pre_b["mining"]),
                )
                self.seats[a]["board"]["gold"] = int(self.seats[a]["board"]["gold"]) + int(pre_a["income"]) + kill_a + lc_a
                self.seats[b]["board"]["gold"] = int(self.seats[b]["board"]["gold"]) + int(pre_b["income"]) + kill_b + lc_b
            else:
                win_field = sum(
                    int((self.content.ships.get(str(p["ship_id"])) or {}).get("cost") or 0)
                    for p in pieces[winner]
                    if p.get("slot") == "field"
                )
                win_pre = pre_a if winner == a else pre_b
                lose_pre = pre_b if winner == a else pre_a
                lose_sid = loser
                lc = loss_comp(
                    econ,
                    loss_streak=int(self.seats[loser]["board"]["loss_streak"]),
                    winner_income=int(win_pre["income"]),
                    winner_field_cost=win_field,
                    titan_pvp=tp,
                    loser_core=int(lose_pre["base"]) + int(lose_pre["interest"]) + int(lose_pre["mining"]),
                )
                self.seats[a]["board"]["gold"] = int(self.seats[a]["board"]["gold"]) + int(pre_a["income"]) + kill_a
                self.seats[b]["board"]["gold"] = int(self.seats[b]["board"]["gold"]) + int(pre_b["income"]) + kill_b
                if lose_sid == a:
                    self.seats[a]["board"]["gold"] = int(self.seats[a]["board"]["gold"]) + lc
                else:
                    self.seats[b]["board"]["gold"] = int(self.seats[b]["board"]["gold"]) + lc
            grant_exp(econ, self.seats[a]["board"], int(econ.get("base_exp_income") or 4))
            grant_exp(econ, self.seats[b]["board"], int(econ.get("base_exp_income") or 4))
            self.seats[a]["gold"] = int(self.seats[a]["board"]["gold"])
            self.seats[b]["gold"] = int(self.seats[b]["board"]["gold"])
            dmg_t = _scaled_titan_loss(self.content, self.cfg)
            if draw:
                for sid in (a, b):
                    self.seats[sid]["titan_hp"] = max(0.0, self.seats[sid]["titan_hp"] - dmg_t)
                    if self.seats[sid]["titan_hp"] <= 0:
                        self.seats[sid]["alive"] = False
            else:
                self.seats[loser]["titan_hp"] = max(0.0, self.seats[loser]["titan_hp"] - dmg_t)
                if self.seats[loser]["titan_hp"] <= 0:
                    self.seats[loser]["alive"] = False
            lc_note = "draw" if draw else str(locals().get("lc", ""))
            self.diag.write(
                f"econ gen={self.gen_i} L={self.league_i} r={rnd+1} {a}:base{pre_a['base']}+int{pre_a['interest']}+win{pre_a['win']}+stk{pre_a['streak']}+mine{pre_a['mining']}+kill{kill_a} "
                f"{b}:base{pre_b['base']}+int{pre_b['interest']}+win{pre_b['win']}+stk{pre_b['streak']}+mine{pre_b['mining']}+kill{kill_b} lc={lc_note}"
            )
            if self.nets is not None:
                ta, tb = traces.get(a) or {}, traces.get(b) or {}
                dmg_a = float(row_a.get("rank_hint") or 0)
                dmg_b = float(row_b.get("rank_hint") or 0)

                def _wipe(opp: dict) -> bool:
                    manned = [s for s in (opp.get("ships") or []) if not s.get("is_unmanned")]
                    return bool(manned) and not any(s.get("survived") for s in manned)

                def _bs(row: dict) -> float:
                    ks = row.get("kill_calendar") or []
                    if not ks:
                        return 90.0
                    return max(90.0, max(float(k.get("t") or 0.0) for k in ks))

                credit_a = merge_ship_equip_credit(
                    self.content, pieces[a], unit_credit(self.content, row_a, _bs(row_a), self.nets.collab)
                )
                credit_b = merge_ship_equip_credit(
                    self.content, pieces[b], unit_credit(self.content, row_b, _bs(row_b), self.nets.collab)
                )
                src = str(getattr(self, "train_source", None) or self.security_mode or "natural")
                sw = source_weight(src)
                if not sa.get("frozen"):
                    apply_genome_delta(
                        sa["genome"], str(sa.get("titan") or ""), credit_a, a_won, content=self.content, source=src
                    )
                if not sb.get("frozen"):
                    apply_genome_delta(
                        sb["genome"], str(sb.get("titan") or ""), credit_b, b_won, content=self.content, source=src
                    )
                ema_blend(self.nets.axis_ema, credit_a)
                ema_blend(self.nets.axis_ema, credit_b)
                adv_a = pair_advantage(
                    self.nets.collab,
                    won=bool(row_a["won"]),
                    draw=draw,
                    dmg_self=dmg_a,
                    dmg_enemy=dmg_b,
                    gold_self=float(int(pre_a["income"]) + kill_a),
                    gold_enemy=float(int(pre_b["income"]) + kill_b),
                    d_hp=0.0 if (row_a["won"] and not draw) else -1.0,
                    pop_self=float((ta or {}).get("field_cost") or 0),
                    pop_enemy=float((tb or {}).get("field_cost") or 0),
                    first_kill=first_kill_s(row_a.get("kill_calendar") or [], row_a.get("ships") or []),
                    wipe=_wipe(row_b),
                    eval_delta=eval_valence_delta(self.content, row_a, pieces[a], a_won),
                )
                adv_b = pair_advantage(
                    self.nets.collab,
                    won=bool(row_b["won"]),
                    draw=draw,
                    dmg_self=dmg_b,
                    dmg_enemy=dmg_a,
                    gold_self=float(int(pre_b["income"]) + kill_b),
                    gold_enemy=float(int(pre_a["income"]) + kill_a),
                    d_hp=0.0 if (row_b["won"] and not draw) else -1.0,
                    pop_self=float((tb or {}).get("field_cost") or 0),
                    pop_enemy=float((ta or {}).get("field_cost") or 0),
                    first_kill=first_kill_s(row_b.get("kill_calendar") or [], row_b.get("ships") or []),
                    wipe=_wipe(row_a),
                    eval_delta=eval_valence_delta(self.content, row_b, pieces[b], b_won),
                )
                sw = source_weight(src)
                if ta:
                    ta = dict(ta)
                    ta["credit"] = credit_a
                    self._remember_trace(sa, ta, adv_a, sw)
                if tb:
                    tb = dict(tb)
                    tb["credit"] = credit_b
                    self._remember_trace(sb, tb, adv_b, sw)
            for sid, oid, won, row_s, row_o in ((a, b, a_won, row_a, row_b), (b, a, b_won, row_b, row_a)):
                mem = self.seats[sid].setdefault("memory", new_memory())
                note_fight(mem, oid, _cost(self.content, pieces[oid]))
                mem.setdefault("rounds", []).append(
                    round_summary(
                        self.content,
                        me_id=sid,
                        opp_id=oid,
                        me_pieces=pieces[sid],
                        opp_pieces=pieces[oid],
                        won=0.5 if draw else (1.0 if won else 0.0),
                        dmg_self=float(row_s.get("rank_hint") or 0),
                        dmg_enemy=float(row_o.get("rank_hint") or 0),
                        bye=False,
                    )
                )
            win_label = "draw" if draw else str(winner)
            self.diag.write(
                f"fight gen={self.gen_i} L={self.league_i} r={rnd+1} {a}vs{b} win={win_label} "
                f"backend={pack.get('backend')} reason={pack.get('end_reason') or '-'} sim={float(pack.get('sim_s') or 0):.0f}"
            )
            self.replay.table_snapshot(
                round_i=rnd,
                seat_a=sa,
                seat_b=sb,
                pieces_a=pieces[a],
                pieces_b=pieces[b],
                gold_a0=gold_before[a],
                gold_b0=gold_before[b],
                result={
                    "winner": None if draw else winner,
                    "loser": None if draw else loser,
                    "draw": draw,
                    "loser_hp": self.seats[a]["titan_hp"] if draw else self.seats[loser]["titan_hp"],
                    "eliminated": False if draw else (not self.seats[loser]["alive"]),
                },
            )
            self._dump_compare_snapshot(ctx, a, b, pack, winner if not draw else -1, loser if not draw else -1)
        pve_jobs = ctx.get("pve_jobs") or []
        for i, job in enumerate(pve_jobs):
            pi = n_pvp + i
            if pi >= len(packs):
                break
            self._apply_pve_pack(ctx, job, packs[pi])

    def _apply_pve_pack(self, ctx: dict, job: dict, pack: dict) -> None:
        rnd = int(ctx["rnd"])
        sid = int(job["seat_a"])
        s = self.seats[sid]
        row_a, row_b = pack["seats"][0], pack["seats"][1]
        task = str(job.get("pve_task") or "pve_eliminate")
        won = pve_success(task=task, row_player=row_a, row_creep=row_b, freighter_id=str(job.get("freighter_id") or ""))
        if won:
            s["wins"] += 1
            s["board"]["win_streak"] = int(s["board"].get("win_streak") or 0) + 1
            s["board"]["loss_streak"] = 0
        else:
            s["board"]["win_streak"] = 0
        pieces = ctx["pieces"]
        gold_before = ctx["gold_before"]
        units_a = combat_units(self.content, pieces[sid], row_a)
        units_b = combat_units(self.content, [], row_b)
        mine_a = mining_gold_from_survivors(self.content, units_a)
        kill_a = kill_gold_of(self.content, units_b)
        econ = self.content.economy
        pre_a = round_income_pre(
            econ,
            gold_ref=int(gold_before[sid]) + kill_a,
            round_i=rnd,
            won=won,
            win_streak=int(s["board"]["win_streak"]),
            mining_g=mine_a,
        )
        s["board"]["gold"] = int(s["board"]["gold"]) + int(pre_a["income"]) + kill_a
        grant_exp(econ, s["board"], int(econ.get("base_exp_income") or 4))
        s["gold"] = int(s["board"]["gold"])
        traces = ctx.get("traces") or {}
        ta = traces.get(sid) or {}
        if self.nets is not None and ta and not s.get("frozen"):
            credit_a = merge_ship_equip_credit(
                self.content, pieces[sid], unit_credit(self.content, row_a, 90.0, self.nets.collab)
            )
            src = str(getattr(self, "train_source", None) or self.security_mode or "natural")
            apply_genome_delta(
                s["genome"], str(s.get("titan") or ""), credit_a, won, content=self.content, source=src
            )
            ema_blend(self.nets.axis_ema, credit_a)
            adv_a = pair_advantage(
                self.nets.collab,
                won=won,
                draw=False,
                dmg_self=float(row_a.get("rank_hint") or 0),
                dmg_enemy=float(row_b.get("rank_hint") or 0),
                gold_self=float(int(pre_a["income"]) + kill_a),
                gold_enemy=0.0,
                d_hp=0.0,
                pop_self=float((ta or {}).get("field_cost") or 0),
                pop_enemy=0.0,
                first_kill=first_kill_s(row_a.get("kill_calendar") or [], row_a.get("ships") or []),
                wipe=won and task == "pve_eliminate",
                eval_delta=eval_valence_delta(self.content, row_a, pieces[sid], won),
            )
            ta = dict(ta)
            ta["credit"] = credit_a
            self._remember_trace(s, ta, adv_a, source_weight(src))
        mem = s.setdefault("memory", new_memory())
        mem.setdefault("rounds", []).append(
            round_summary(
                self.content,
                me_id=sid,
                opp_id=None,
                me_pieces=pieces[sid],
                opp_pieces=None,
                won=1.0 if won else 0.0,
                dmg_self=float(row_a.get("rank_hint") or 0),
                dmg_enemy=float(row_b.get("rank_hint") or 0),
                bye=False,
            )
        )
        self.diag.write(
            f"pve gen={self.gen_i} L={self.league_i} r={rnd+1} seat={sid} task={task} ok={won} "
            f"kill={kill_a} hp={s['titan_hp']:.0f} backend={pack.get('backend')}"
        )
        creep_ids = [str(p.get("ship_id") or "") for p in (job.get("pos_b") or [])]
        # Text replay: one league keeps full board layouts; all leagues go into match_checkpoint.json.
        if self.league_i == 0:
            self.replay.pve_snapshot(
                round_i=rnd,
                seat=s,
                pieces=list(pieces.get(sid) or []),
                gold0=int(gold_before[sid]),
                task=task,
                won=won,
                freighter_id=str(job.get("freighter_id") or ""),
                creep_ids=creep_ids,
            )

    def _dump_compare_snapshot(self, ctx: dict, a: int, b: int, pack: dict, winner: int, loser: int) -> None:
        jobs = ctx.get("jobs") or []
        job = next((j for j in jobs if j.get("seat_a") == a and j.get("seat_b") == b), None)
        if not job:
            return
        fa = sum(1 for p in job.get("pos_a") or [] if p)
        fb = sum(1 for p in job.get("pos_b") or [] if p)
        if fa <= 0 or fb <= 0:
            return
        out_dir = ROOT / "samples" / "compare_snapshots"
        out_dir.mkdir(parents=True, exist_ok=True)
        if len(list(out_dir.glob("*.json"))) >= 40:
            return
        row_a, row_b = pack["seats"][0], pack["seats"][1]
        blob = {
            "job": {
                "fleet_a": job["fleet_a"],
                "fleet_b": job["fleet_b"],
                "pos_a": job.get("pos_a"),
                "pos_b": job.get("pos_b"),
                "titan_a": job.get("titan_a"),
                "titan_b": job.get("titan_b"),
                "seed": job.get("seed"),
                "match_id": job.get("match_id"),
                "round_i": job.get("round_i"),
                "seat_a": a,
                "seat_b": b,
            },
            "farm": {
                "backend": pack.get("backend"),
                "winner_seat": winner if winner >= 0 else None,
                "a_won": bool(row_a.get("won")),
                "b_won": bool(row_b.get("won")),
                "draw": (not row_a.get("won")) and (not row_b.get("won")),
                "titan_hp_a": row_a.get("titan_hp"),
                "titan_hp_b": row_b.get("titan_hp"),
                "rank_hint_a": row_a.get("rank_hint"),
                "rank_hint_b": row_b.get("rank_hint"),
                "ships_a": row_a.get("ships"),
                "ships_b": row_b.get("ships"),
            },
        }
        name = f"{job.get('match_id','m')}_r{int(job.get('round_i') or 0)+1}_{a}v{b}.json"
        (out_dir / name).write_text(json.dumps(blob, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def finalize(self) -> dict:
        self.done = True
        ranked = sorted(self.seats, key=lambda s: _elite_key(s, self.content.titan_pvp))
        elites = ranked[: self.keep]
        elite_ids = {e["seat_id"] for e in elites}
        if self.nets is not None:
            for s in self.seats:
                if s.get("frozen"):
                    continue
                lps = [x for x in (s.get("titan_lps") or []) if x is not None]
                if not lps or self.nets._live_lp(lps[0]) is None:
                    continue
                adv = 0.25 if s["seat_id"] in elite_ids else -0.05
                self.nets.remember({"ops_lp": lps[0] * 0.0, "adv_hat": lps[0] * 0.0, "titan_lps": lps, "shop_lps": [], "fit_lps": [], "place_lps": []}, adv)
        self.replay.ranking(ranked, self.keep)
        self.diag.write(f"elite gen={self.gen_i} L={self.league_i} {[e['seat_id'] for e in elites]}")
        return {"seats": self.seats, "ranked": ranked, "elites": elites, "seen": self.seen, "draft": self.draft}


def run_generation(
    *,
    content: Content,
    cfg: dict,
    kernel: BattleKernel | GpuBattleKernel,
    use_gpu: bool,
    infer,
    base: dict,
    seed: int,
    gen_i: int,
    diag: CappedLog,
    replay: ReplayWriter,
    replay_log: CappedLog,
    nets: FourNetPack | None = None,
    resume_blob: dict | None = None,
) -> dict:
    titans = list(cfg.get("titan_ids") or ["amarr", "caldari", "gallente", "minmatar", "angel"])
    stances = list(cfg.get("stance_ids") or ["economy", "offense", "logistics", "speed_control", "formation"])
    keep = int(cfg.get("keep_top", 3))
    mode = str(cfg.get("_security_mode") or "nullsec")
    if mode == "lowsec":
        keep = 1
        parallel = max(1, int(cfg.get("lowsec_tables") or cfg.get("_parallel") or 40))
        cfg["_parallel"] = parallel
    else:
        parallel = max(1, int(cfg.get("_parallel") or cfg.get("matches_per_batch") or cfg.get("cuda_parallel_matches") or 1))
    leagues = [
        MatchLeague(
            content=content,
            cfg=cfg,
            infer=infer,
            use_gpu=use_gpu,
            base=base,
            seed=int(seed) + li * 7919,
            gen_i=gen_i,
            league_i=li,
            diag=diag,
            replay=replay,
            replay_log=replay_log,
            nets=nets,
        )
        for li in range(parallel)
    ]
    samples = ROOT / "samples"
    resume_rnd = 0
    if resume_blob and resume_blob.get("gen_i") is not None and int(resume_blob.get("gen_i")) == int(gen_i):
        resume_rnd = restore_leagues_from_checkpoint(leagues, resume_blob)
        if str(resume_blob.get("kind") or "") == "end":
            for L in leagues:
                L.done = True
            print(f"resume end checkpoint gen={gen_i} mode={mode} tables={len(leagues)} finalize ranking 中文UTF8", flush=True)
            diag.write(f"resume end_checkpoint gen={gen_i} mode={mode} leagues={len(leagues)}")
        else:
            print(
                f"resume mid_match gen={gen_i} mode={mode} continue_round={resume_rnd + 1} "
                f"tables={len(leagues)} 中文UTF8",
                flush=True,
            )
            diag.write(f"resume mid_match gen={gen_i} mode={mode} rnd={resume_rnd} leagues={len(leagues)}")
    else:
        from eveac_ai.capital_bank import apply_fuse_to_match_league

        frac = float((cfg.get("curriculum") or {}).get("capital_frac") or 0.0)
        n_cap = int(round(len(leagues) * max(0.0, min(1.0, frac))))
        cap_rng = random.Random(int(seed) + 17)
        for L in leagues[max(0, len(leagues) - n_cap) :]:
            apply_fuse_to_match_league(content, L, samples, cap_rng, mode)
            L.diag.write(f"capital_bank gen={gen_i} L={L.league_i} fused=1")
    rnd = resume_rnd
    wave_times: list[float] = []
    sim_agg: dict = {}
    while True:
        jobs: list[dict] = []
        slices: list[tuple[MatchLeague, dict, int, int]] = []
        for L in leagues:
            chunk, ctx = L.collect_jobs(rnd)
            if not ctx:
                continue
            start = len(jobs)
            jobs.extend(chunk)
            slices.append((L, ctx, start, len(chunk)))
        if not jobs:
            break
        # region agent log
        n_flag_h = n_cyno_f = n_jump = n_jobs_hold = n_hold_units = 0
        n_alive = 0
        for L, *_rest in slices:
            ships_m = L.content.ships
            for s in L.seats:
                if not s.get("alive"):
                    continue
                n_alive += 1
                pcs = (s.get("board") or {}).get("pieces") or []
                fh = any(is_cyno_flagship(ships_m.get(str(p.get("ship_id")))) and p.get("slot") == "hangar" for p in pcs)
                cf = any(is_covert_cyno(ships_m.get(str(p.get("ship_id")))) and p.get("slot") == "field" for p in pcs)
                if fh:
                    n_flag_h += 1
                if cf:
                    n_cyno_f += 1
                if fh and cf:
                    n_jump += 1
        for j in jobs:
            holds = [p for p in (j.get("pos_a") or []) + (j.get("pos_b") or []) if p.get("cyno_hold")]
            n_hold_units += len(holds)
            if holds:
                n_jobs_hold += 1
        _agent_dbg(
            "H",
            "match20.py:wave_flagships",
            "hangar flag vs cyno field vs fight inject",
            {
                "rnd": rnd + 1,
                "n_alive_seats": n_alive,
                "n_jobs": len(jobs),
                "flag_hangar_seats": n_flag_h,
                "cyno_field_seats": n_cyno_f,
                "would_jump_seats": n_jump,
                "jobs_with_cyno_hold": n_jobs_hold,
                "cyno_hold_units": n_hold_units,
            },
        )
        # endregion
        t_wave = time.time()
        packs, route = route_fight_batch(jobs, kernel=kernel, use_gpu=use_gpu, cfg=cfg)
        wins = draws = 0
        draw_cap = draw_no_off = draw_empty = draw_nodmg = draw_wall = pvp_n = 0
        for job, pack in zip(jobs, packs):
            seats = pack.get("seats") or []
            if len(seats) < 2:
                continue
            if job.get("pve_task"):
                ok = pve_success(
                    task=str(job.get("pve_task") or ""),
                    row_player=seats[0],
                    row_creep=seats[1],
                    freighter_id=str(job.get("freighter_id") or ""),
                )
                if ok:
                    wins += 1
                continue
            pvp_n += 1
            a_won = bool(seats[0].get("won"))
            b_won = bool(seats[1].get("won"))
            if a_won or b_won:
                wins += 1
            else:
                draws += 1
                reason = str(pack.get("end_reason") or "")
                sim = float(pack.get("sim_s") or 0.0)
                dmg = sum(float(m.get("dmg_out") or 0) for m in (seats[0].get("ships") or []) + (seats[1].get("ships") or []))
                if dmg < 1.0:
                    draw_nodmg += 1
                if reason == "cap" or sim >= 899.0:
                    draw_cap += 1
                elif reason == "no_off":
                    draw_no_off += 1
                elif reason == "empty":
                    draw_empty += 1
                elif reason == "wall":
                    draw_wall += 1
                elif sim >= 899.0:
                    draw_cap += 1
                else:
                    draw_no_off += 1
        # region agent log
        hold_fired = hold_idle = hold_sim_lt90 = 0
        for job, pack in zip(jobs, packs):
            hold_ids = {
                str(p.get("ship_id"))
                for p in (job.get("pos_a") or []) + (job.get("pos_b") or [])
                if p.get("cyno_hold")
            }
            if not hold_ids:
                continue
            sim = float(pack.get("sim_s") or 0.0)
            if sim < 90.0:
                hold_sim_lt90 += 1
            for seat in pack.get("seats") or []:
                for m in seat.get("ships") or []:
                    if str(m.get("ship_id")) not in hold_ids:
                        continue
                    if float(m.get("dmg_out") or 0) > 0 or float(m.get("lock_s") or 0) > 1.0:
                        hold_fired += 1
                    else:
                        hold_idle += 1
        _agent_dbg(
            "I",
            "match20.py:wave_flag_combat",
            "injected flagship fired vs idle",
            {
                "rnd": rnd + 1,
                "hold_fired": hold_fired,
                "hold_idle": hold_idle,
                "jobs_sim_lt90_with_hold": hold_sim_lt90,
            },
        )
        kinds_now = "+".join(sorted({"pve" if j.get("pve_task") else "pvp" for j in jobs})) or "pvp"
        _agent_dbg(
            "L",
            "match20.py:wave_draws",
            "pvp draw breakdown vs architecture",
            {
                "rnd": rnd + 1,
                "kind": kinds_now,
                "route": route,
                "pvp_n": pvp_n,
                "pvp_decided": max(0, pvp_n - draws),
                "draw": draws,
                "draw_rate": round(draws / max(pvp_n, 1), 3),
                "draw_cap": draw_cap,
                "draw_no_off": draw_no_off,
                "draw_empty": draw_empty,
                "draw_wall": draw_wall,
                "draw_nodmg": draw_nodmg,
                "n_cap_batch": (getattr(kernel, "last_timing", None) or {}).get("n_cap"),
            },
        )
        # endregion
        for L, ctx, start, n in slices:
            L.apply_packs(ctx, packs[start : start + n])
        loss = 0.0
        if nets is not None:
            for L, ctx, start, n in slices:
                L.flush_dead_traces()
            # region agent log
            buf_n = {
                "ops": len(getattr(nets, "_ops_buf", []) or []),
                "shop": len(getattr(nets, "_shop_buf", []) or []),
                "fit": len(getattr(nets, "_fit_buf", []) or []),
                "place": len(getattr(nets, "_place_buf", []) or []),
                "titan": len(getattr(nets, "_titan_buf", []) or []),
            }
            n_paired = sum(1 for L, *_ in slices if getattr(L, "_trace_paired", False))
            n_prev = sum(1 for L, *_ in slices for s in L.seats if s.get("prev_trace"))
            n_tr = sum(1 for L, ctx, *_ in slices for t in (ctx.get("traces") or {}).values() if t)
            # endregion
            loss = nets.backward_step()
            diag.write(f"nets gen={gen_i} r={rnd+1} tables={len(jobs)} loss={loss:.4f}")
            # region agent log
            kinds_now = "+".join(sorted({"pve" if j.get("pve_task") else "pvp" for j in jobs})) or "pvp"
            _agent_dbg(
                "G",
                "match20.py:wave_train",
                "pve/pvp loss vs pair-stash",
                {
                    "rnd": rnd + 1,
                    "kind": kinds_now,
                    "odd_round": (rnd + 1) % 2 == 1,
                    "loss": round(float(loss), 6),
                    "n_paired_leagues": n_paired,
                    "n_prev_trace": n_prev,
                    "n_traces": n_tr,
                    "buf_before": buf_n,
                    "backward": getattr(nets, "last_backward", None) or {},
                    "arch_expect_stash": False,
                    "arch_expect_loss0": False,
                    "arch_every_table_back": True,
                },
            )
            # endregion
        alive_seats = [s for L in leagues for s in L.seats if s.get("alive")]
        alive = len(alive_seats)
        hp_min = min((float(s["titan_hp"]) for s in alive_seats), default=0.0)
        wave_s = time.time() - t_wave
        wave_times.append(wave_s)
        sim_t = float(getattr(kernel, "last_sim_s", 0.0) or 0.0)
        # region agent log
        cpu_k = kernel.cpu if isinstance(kernel, GpuBattleKernel) else kernel
        printed = round(sim_t)
        _agent_dbg(
            "C",
            "match20.py:wave_print",
            "stdout sim field vs cpu/gpu clocks",
            {
                "route": route,
                "tables": len(jobs),
                "wave_s": round(wave_s, 3),
                "printed_sim_0f": printed,
                "wrapper_sim": sim_t,
                "cpu_sim": float(getattr(cpu_k, "last_sim_s", 0.0) or 0.0),
                "timing": getattr(kernel, "last_timing", None) or {},
            },
        )
        # endregion
        kinds = "+".join(sorted({"pve" if j.get("pve_task") else "pvp" for j in jobs})) or "pvp"
        mode = str(cfg.get("_security_mode") or "nullsec")
        sim_sum = summary_from_timing(
            getattr(kernel, "last_timing", None),
            interval_s=float(cfg.get("stats_interval_s") or 10),
        )
        add_summaries(sim_agg, sim_sum)
        extra = format_round_stats(sim_sum)
        train = "back"
        print(
            f"gen={gen_i} mode={mode} route={route} round={rnd+1} kind={kinds} tables={len(jobs)} leagues={len(slices)} "
            f"decided={wins} draw={draws} alive={alive} hp_min={hp_min:.0f} loss={loss:.4f} train={train} "
            f"{extra}".rstrip(),
            flush=True,
        )
        for L in leagues:
            if not L.done:
                L.replay.round_marker(match_id=L.match_id, round_i=rnd, kind=kinds, alive=sum(1 for s in L.seats if s.get("alive")))
        if _should_write_checkpoint(cfg, rnd + 1):
            _write_match_checkpoint(
                ROOT / "samples",
                gen_i=gen_i,
                round_done=rnd,
                seed=seed,
                leagues=leagues,
                kinds=kinds,
                nets=nets,
                train_meta={
                    "phase": str(cfg.get("_phase") or ("mass" if int(cfg.get("_parallel") or 1) > 1 else "fast")),
                    "parallel": int(cfg.get("_parallel") or 1),
                    "catalog": len(spend_catalog(content.ships, content.equip_meta)),
                    "security_mode": mode,
                },
                security_mode=mode,
                ingest_bank=True,
            )
        rnd += 1
    outs = [L.finalize() for L in leagues]
    _write_match_checkpoint(
        ROOT / "samples",
        gen_i=gen_i,
        round_done=max(0, rnd - 1),
        seed=seed,
        leagues=leagues,
        kinds="end",
        nets=nets,
        train_meta={
            "phase": str(cfg.get("_phase") or ""),
            "parallel": int(cfg.get("_parallel") or 1),
            "security_mode": str(cfg.get("_security_mode") or "nullsec"),
        },
        security_mode=str(cfg.get("_security_mode") or "nullsec"),
        ingest_bank=True,
    )
    if nets is not None:
        for L in leagues:
            L.flush_pending_traces()
        loss = nets.backward_step()
        diag.write(f"nets gen={gen_i} titan_end loss={loss:.4f}")
    pool: list[dict] = []
    seen: set[str] = set()
    nat_stats = {"xp_buys": 0.0, "prepares": 0.0, "field_full": 0.0, "skipped_xp": 0.0, "level_sum": 0.0, "level15": 0.0, "cyno_field": 0.0, "flag_hangar": 0.0, "flag_jump": 0.0, "n": 0.0}
    cap_stats = dict(nat_stats)
    for L, o in zip(leagues, outs):
        st = cap_stats if getattr(L, "train_source", "") == "capital_bank" else nat_stats
        st["n"] += 1
        for k, v in (getattr(L, "stats", None) or {}).items():
            if k in st:
                st[k] += float(v)
        if getattr(L, "train_source", "") == "capital_bank":
            continue
        seen |= {str(x) for x in o["seen"]}
        for s in o["ranked"]:
            if s.get("scripted"):
                continue
            row = dict(s)
            row["seat_id"] = f"L{L.league_i}:{s['seat_id']}"
            pool.append(row)

    fsm_agg = empty_fsm_stats()
    for L, o in zip(leagues, outs):
        add_fsm_stats(fsm_agg, league_fsm_stats(L, o))

    def _rate(st: dict, num: str, den: str) -> str:
        d = st.get(den) or 0.0
        if d <= 0:
            return "na"
        return f"{st.get(num, 0.0) / d:.3f}"

    print(
        f"pop. natural tables={int(nat_stats['n'])} xp_buy={_rate(nat_stats,'xp_buys','prepares')} "
        f"skip_full={_rate(nat_stats,'skipped_xp','field_full')} lv15={_rate(nat_stats,'level15','prepares')} "
        f"mean_lv={_rate(nat_stats,'level_sum','prepares')} cyno_field={_rate(nat_stats,'cyno_field','prepares')} "
        f"jump={_rate(nat_stats,'flag_jump','flag_hangar')} fsm={int(fsm_agg['seats'])} "
        f"| capital_bank tables={int(cap_stats['n'])} "
        f"jump={_rate(cap_stats,'flag_jump','flag_hangar')} cyno_field={_rate(cap_stats,'cyno_field','prepares')}",
        flush=True,
    )
    print(format_gen_stats(sim_agg, gen=gen_i), flush=True)
    diag.write(
        f"pop gen={gen_i} nat={nat_stats} cap={cap_stats} fsm={fsm_agg} sim={sim_agg}"
    )
    if not pool:
        for L, o in zip(leagues, outs):
            seen |= {str(x) for x in o["seen"]}
            for s in o["ranked"]:
                if s.get("scripted"):
                    continue
                row = dict(s)
                row["seat_id"] = f"L{L.league_i}:{s['seat_id']}"
                pool.append(row)
    ranked = sorted(pool, key=lambda s: _elite_key(s, content.titan_pvp))
    elites = ranked[:keep]
    export = blend_genomes([e["genome"] for e in elites], titans, stances, content=content)
    export["content_rev"] = content.rev
    diag.write(f"elite gen={gen_i} pooled={[e['seat_id'] for e in elites]} leagues={parallel} jobs_wave=cuda_batch")
    return {
        "seats": pool,
        "ranked": ranked,
        "elites": elites,
        "export": export,
        "seen": seen,
        "draft": outs[0]["draft"],
        "wave_s": wave_times,
        "fsm": fsm_agg,
    }


def run_match(**kwargs) -> dict:
    return run_generation(**kwargs)


def _ranking_names(content: Content) -> dict[str, str]:
    catalog = spend_catalog(content.ships, content.equip_meta)
    names: dict[str, str] = {}
    for kind, iid in catalog:
        key = item_key(kind, iid)
        if kind == "ship":
            names[key] = ship_label(content, iid)
            names[iid] = names[key]
        else:
            names[key] = str((content.equip_meta.get(iid) or {}).get("name") or iid)
            names[iid] = names[key]
    return names


def _write_ranking(
    samples: Path,
    content: Content,
    export: dict,
    seen: set[str],
    *,
    security_mode: str = "nullsec",
    fsm_line: str | None = None,
) -> dict:
    catalog = spend_catalog(content.ships, content.equip_meta)
    names = {sid: ship_label(content, sid) for kind, sid in catalog if kind == "ship"}
    eq_names = {eid: str((content.equip_meta.get(eid) or {}).get("name") or eid) for kind, eid in catalog if kind == "equip"}
    table = quantize_ships(
        export,
        [sid for kind, sid in catalog if kind == "ship"],
        seen_ids=seen,
        content_rev=content.rev,
        ship_names=names,
        hulls=content.ships,
        equip_meta=content.equip_meta,
        equip_names=eq_names,
    )
    table["security_mode"] = str(security_mode or "nullsec")
    samples.mkdir(parents=True, exist_ok=True)
    (samples / "weights_table.json").write_text(json.dumps(table, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    mode = "lowsec" if security_mode == "lowsec" else "nullsec"
    (samples / f"weights_table_{mode}.json").write_text(
        json.dumps(table, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    body = format_ranking_table(table, names)
    if fsm_line:
        body = f"{fsm_line}\n\n{body}"
    (samples / "ranking_ships.txt").write_text(body, encoding="utf-8")
    return table


def _write_session(samples: Path, *, next_batch: int, reason: str, wall_s: float, extra: dict | None = None) -> None:
    blob = {
        "next_batch": int(next_batch),
        "reason": str(reason),
        "wall_s": round(float(wall_s), 1),
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if extra:
        blob.update(extra)
    (samples / "session.json").write_text(json.dumps(blob, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _json_safe(obj: object) -> object:
    """Drop / convert torch tensors and other non-JSON values for mid-match checkpoints."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, (set, frozenset)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    mod = type(obj).__module__ or ""
    name = type(obj).__name__
    if "torch" in mod and name == "Tensor":
        try:
            return obj.detach().cpu().tolist()  # type: ignore[attr-defined]
        except Exception:
            return None
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    return str(obj)


def _seat_checkpoint(s: dict) -> dict:
    """Machine-readable seat for mid-match resume (layout + economy + genome)."""
    board = s.get("board") or {}
    return {
        "seat_id": int(s["seat_id"]),
        "titan": s.get("titan"),
        "titan_round1": s.get("titan_round1"),
        "titan_hp": float(s.get("titan_hp") or 0),
        "wins": int(s.get("wins") or 0),
        "losses": int(s.get("losses") or 0),
        "alive": bool(s.get("alive")),
        "gold": int(s.get("gold") or board.get("gold") or 0),
        "fleet": list(s.get("fleet") or []),
        "board": {
            "gold": int(board.get("gold") or 0),
            "level": int(board.get("level") or 1),
            "xp": int(board.get("xp") or 0),
            "win_streak": int(board.get("win_streak") or 0),
            "loss_streak": int(board.get("loss_streak") or 0),
            "bag": list(board.get("bag") or []),
            "shop": list(board.get("shop") or []),
            "token": int(board.get("token") or 1),
            "pieces": _json_safe(list(board.get("pieces") or [])),
        },
        "genome": _json_safe(s.get("genome")),
        "memory": _json_safe(s.get("memory")),
        "titan_lps": _json_safe(s.get("titan_lps")),
        "frozen": bool(s.get("frozen")),
        "scripted": str(s.get("scripted") or ""),
    }


def _write_match_checkpoint(
    samples: Path,
    *,
    gen_i: int,
    round_done: int,
    seed: int,
    leagues: list,
    kinds: str,
    nets: FourNetPack | None,
    train_meta: dict | None = None,
    security_mode: str = "nullsec",
    ingest_bank: bool = False,
) -> None:
    """Overwrite mid-match checkpoint (throttled by caller). Optionally ingest StateBank."""
    leagues_blob = []
    for L in leagues:
        leagues_blob.append(
            {
                "league_i": int(L.league_i),
                "match_id": L.match_id,
                "seed": int(L.seed),
                "done": bool(L.done),
                "round_kind": str(getattr(L, "_round_kind", "") or ""),
                "security_mode": str(getattr(L, "security_mode", security_mode) or security_mode),
                "train_source": str(getattr(L, "train_source", "") or security_mode),
                "draft": _json_safe(L.draft),
                "seats": [_seat_checkpoint(s) for s in L.seats],
            }
        )
    blob = {
        "schema_ver": "1",
        "purpose": "mid_match_resume",
        "security_mode": str(security_mode or "nullsec"),
        "gen_i": int(gen_i),
        "round_done": int(round_done),
        "next_round": int(round_done) + 1,
        "match_seed_base": int(seed),
        "kind": str(kinds),
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "leagues": leagues_blob,
        "nets_dir": "samples/nets",
        "note": "Resume uses round_done+1 as 0-based rnd. Nets: behavior.nets.pt mid-save if present.",
    }
    samples.mkdir(parents=True, exist_ok=True)
    path = samples / "match_checkpoint.json"
    text = json.dumps(_json_safe(blob), ensure_ascii=False, indent=2) + "\n"
    path.write_text(text, encoding="utf-8")
    snap_dir = samples / "logs" / "match_checkpoints"
    snap_dir.mkdir(parents=True, exist_ok=True)
    snap = snap_dir / f"gen{gen_i}_r{round_done + 1}.json"
    snap.write_text(json.dumps(_json_safe(blob), ensure_ascii=False) + "\n", encoding="utf-8")
    if nets is not None:
        try:
            nets.save()
        except Exception:
            pass
    if ingest_bank:
        try:
            ingest_checkpoint(samples, blob, mode=str(security_mode or "nullsec"))
        except Exception:
            pass
    prev: dict = {}
    sp = samples / "session.json"
    if sp.is_file():
        try:
            prev = json.loads(sp.read_text(encoding="utf-8")) or {}
        except (json.JSONDecodeError, TypeError, ValueError):
            prev = {}
    meta = dict(train_meta or {})
    extra = {
        "mid_match": True,
        "round_done": int(round_done),
        "next_round": int(round_done) + 1,
        "checkpoint": str(path.name),
        "security_mode": str(security_mode or "nullsec"),
    }
    for k in ("phase", "tu", "catalog", "seen", "parallel"):
        if k in meta:
            extra[k] = meta[k]
        elif k in prev:
            extra[k] = prev[k]
    _write_session(
        samples,
        next_batch=int(gen_i),
        reason="mid_match",
        wall_s=float(prev.get("wall_s") or meta.get("wall_s") or 0.0),
        extra=extra,
    )


def apply_seat_checkpoint(seat: dict, blob: dict) -> None:
    """Overwrite a live seat with mid-match checkpoint fields."""
    board0 = new_seat_board()
    src = dict(blob.get("board") or {})
    board0.update({k: v for k, v in src.items() if k != "shop"})
    if not board0.get("shop_ships"):
        board0["shop_ships"] = list(src.get("shop") or src.get("shop_ships") or [])
    if not board0.get("shop_equips"):
        board0["shop_equips"] = list(src.get("shop_equips") or [])
    seat["titan"] = blob.get("titan") or seat.get("titan")
    seat["titan_round1"] = blob.get("titan_round1")
    seat["titan_hp"] = float(blob.get("titan_hp") or 0)
    seat["wins"] = int(blob.get("wins") or 0)
    seat["losses"] = int(blob.get("losses") or 0)
    seat["alive"] = bool(blob.get("alive"))
    seat["fleet"] = list(blob.get("fleet") or [])
    seat["board"] = board0
    seat["gold"] = int(blob.get("gold") or board0.get("gold") or 0)
    if blob.get("genome") is not None:
        seat["genome"] = blob.get("genome")
    seat["memory"] = hydrate_memory(blob.get("memory"))
    seat["titan_lps"] = blob.get("titan_lps")
    if "frozen" in blob:
        seat["frozen"] = bool(blob.get("frozen"))
    if "scripted" in blob:
        seat["scripted"] = str(blob.get("scripted") or "")
        if seat["scripted"]:
            seat["frozen"] = True


def restore_leagues_from_checkpoint(leagues: list, blob: dict) -> int:
    """Apply checkpoint seats. Missing league_i (old dumps skipped done tables) stay done. Return next 0-based rnd."""
    by_i = {int(getattr(L, "league_i", -1)): L for L in leagues}
    seen: set[int] = set()
    for lb in blob.get("leagues") or []:
        if lb.get("league_i") is None:
            continue
        li = int(lb.get("league_i"))
        L = by_i.get(li)
        if L is None:
            continue
        seen.add(li)
        L.done = bool(lb.get("done"))
        if lb.get("draft") is not None:
            L.draft = lb.get("draft")
        L._round_kind = str(lb.get("round_kind") or "")
        if lb.get("security_mode"):
            L.security_mode = str(lb.get("security_mode"))
        src = str(lb.get("train_source") or L.security_mode or "")
        L.train_source = src
        L.force_pvp = bool(L.security_mode == "lowsec" or src == "capital_bank")
        if lb.get("match_id"):
            L.match_id = str(lb.get("match_id"))
        if lb.get("seed") is not None:
            L.seed = int(lb.get("seed"))
        seats_blob = {}
        for x in lb.get("seats") or []:
            if not isinstance(x, dict) or x.get("seat_id") is None:
                continue
            seats_blob[int(x.get("seat_id"))] = x
        for s in L.seats:
            if s.get("seat_id") is None:
                continue
            row = seats_blob.get(int(s.get("seat_id")))
            if row:
                apply_seat_checkpoint(s, row)
    for L in leagues:
        if int(getattr(L, "league_i", -1)) not in seen:
            L.done = True
    return int(blob.get("round_done") or 0) + 1


def _load_match_checkpoint(samples: Path) -> dict | None:
    p = samples / "match_checkpoint.json"
    if not p.is_file():
        return None
    try:
        blob = json.loads(p.read_text(encoding="utf-8")) or {}
    except (json.JSONDecodeError, TypeError, ValueError, OSError):
        return None
    if not isinstance(blob, dict) or not blob.get("leagues"):
        return None
    return blob


def _mid_match_resume_blob(samples: Path, batch: int) -> dict | None:
    sp = samples / "session.json"
    sess: dict = {}
    if sp.is_file():
        try:
            sess = json.loads(sp.read_text(encoding="utf-8")) or {}
        except (json.JSONDecodeError, TypeError, ValueError, OSError):
            sess = {}
    if not sess.get("mid_match"):
        return None
    if sess.get("next_batch") is None or int(sess.get("next_batch")) != int(batch):
        return None
    blob = _load_match_checkpoint(samples)
    if blob is None:
        return None
    if blob.get("gen_i") is None or int(blob.get("gen_i")) != int(batch):
        return None
    return blob


def _load_seen(samples: Path) -> set[str]:
    p = samples / "seen_spend.json"
    if not p.is_file():
        return set()
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            raw = raw.get("keys") or []
        return {str(x) for x in (raw or [])}
    except (json.JSONDecodeError, TypeError, ValueError):
        return set()


def _save_seen(samples: Path, seen: set[str]) -> None:
    keys = sorted(seen)
    (samples / "seen_spend.json").write_text(json.dumps({"keys": keys}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _tu_n(table: dict) -> int:
    return len(((table.get("tiers") or {}).get("TU") or {}).get("items") or [])


def _apply_train_phase(cfg: dict, kernel: BattleKernel | GpuBattleKernel, nets: FourNetPack | None, *, mass: bool) -> str:
    fast_n = max(1, int(cfg.get("matches_per_batch_fast") or 1))
    mass_n = max(1, int(cfg.get("matches_per_batch") or cfg.get("cuda_parallel_matches") or 8))
    sim_cap = float(kernel.content.match_flow.get("battle_duration_s", 900) or 900)
    kernel.max_s = sim_cap
    if hasattr(kernel, "cpu"):
        kernel.cpu.max_s = sim_cap
    if mass:
        cfg["_parallel"] = mass_n
        cfg["_phase"] = "mass"
        if nets is not None:
            nets.collab["lam_econ"] = 0.5
            nets.collab["beta_shop"] = 1.0
        return "mass"
    cfg["_parallel"] = fast_n
    cfg["_phase"] = "fast"
    if nets is not None:
        nets.collab["lam_econ"] = 1.2
        nets.collab["beta_shop"] = 0.55
    return "fast"


def _load_resume_prior(samples: Path, content: Content) -> tuple[dict, int]:
    genome_path = samples / "behavior.genome.json"
    session_path = samples / "session.json"
    if not genome_path.is_file():
        return load_bootstrap(content=content), 0
    genome = json.loads(genome_path.read_text(encoding="utf-8"))
    genome = scrub_genome_ships(genome, content)
    nxt = 0
    if session_path.is_file():
        try:
            nxt = int((json.loads(session_path.read_text(encoding="utf-8")) or {}).get("next_batch") or 0)
        except (json.JSONDecodeError, TypeError, ValueError):
            nxt = 0
    return genome, max(0, nxt)


def _load_fsm_lifetime(samples: Path) -> dict:
    session_path = samples / "session.json"
    if not session_path.is_file():
        return empty_fsm_stats()
    try:
        blob = json.loads(session_path.read_text(encoding="utf-8")) or {}
    except (json.JSONDecodeError, TypeError, ValueError, OSError):
        return empty_fsm_stats()
    raw = blob.get("fsm_lifetime")
    out = empty_fsm_stats()
    if isinstance(raw, dict):
        add_fsm_stats(out, raw)
    return out


def main(minutes: float | None = None, *, fresh: bool = False) -> None:
    configure_utf8_stdio()
    cfg = load_config()
    content = Content(cfg=cfg)
    samples = ROOT / cfg.get("out_dir", "samples")
    samples.mkdir(parents=True, exist_ok=True)
    lock_path = acquire_farm_lock(samples)
    if lock_path is None:
        sys.exit(2)
    try:
        _main_locked(minutes=minutes, fresh=fresh, cfg=cfg, content=content, samples=samples)
    finally:
        release_farm_lock(lock_path)


def _pick_security_mode(cfg: dict, rng: random.Random) -> str:
    enabled = list(cfg.get("security_modes_enabled") or ["nullsec"])
    enabled = [str(x) for x in enabled if str(x) in ("nullsec", "lowsec")]
    if not enabled:
        enabled = ["nullsec"]
    if "lowsec" in enabled and len(enabled) > 1:
        frac = float(cfg.get("lowsec_frac") or 0.2)
        if rng.random() < frac:
            return "lowsec"
        return "nullsec"
    return enabled[0]


def _main_locked(
    *,
    minutes: float | None,
    fresh: bool,
    cfg: dict,
    content: Content,
    samples: Path,
) -> None:
    if fresh:
        prior = load_bootstrap(content=content)
        batch = 0
    else:
        prior, batch = _load_resume_prior(samples, content)
    infer = resolve_device("auto")
    want_gpu = str(cfg.get("battle_backend") or "").lower() == "gpu"
    use_gpu = want_gpu and infer.kind == "cuda" and infer.torch_device is not None
    try:
        import torch

        torch.set_num_threads(max(1, int(torch.get_num_threads() or 1)))
        if infer.kind == "cuda":
            torch.set_float32_matmul_precision("high")
    except ImportError:
        pass
    sparse = cfg.get("gpu_sparse_dt")
    if sparse is None:
        sparse = True
    kernel: BattleKernel | GpuBattleKernel = (
        GpuBattleKernel(content, infer.torch_device, sparse_dt=bool(sparse)) if use_gpu else BattleKernel(content)
    )
    net_dev = infer.torch_device if infer.kind == "cuda" and infer.torch_device is not None else None
    nets_dir = ROOT / "samples" / "nets"
    nets = FourNetPack(device=net_dev, dir_path=nets_dir)
    log_dir = ROOT / "samples" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    diag = CappedLog(log_dir / "diag.log", kind="diag")
    replay_log = CappedLog(log_dir / "replay.txt", kind="replay")
    replay = ReplayWriter(replay_log, content, content.board)
    budget = None if minutes is None or minutes <= 0 else float(minutes) * 60.0
    stop = {"flag": False}

    def _request_stop(signum, _frame) -> None:
        stop["flag"] = True
        print(f"stop requested signal={signum}; will save after current batch", flush=True)
        diag.write(f"session stop_requested signal={signum}")

    for sig in (signal.SIGINT, signal.SIGTERM, getattr(signal, "SIGBREAK", signal.SIGINT)):
        try:
            signal.signal(sig, _request_stop)
        except (ValueError, OSError):
            pass
    t0 = time.time()
    last: dict | None = None
    start_batch = batch
    rank_prev = None
    fsm_life = empty_fsm_stats() if fresh else _load_fsm_lifetime(samples)
    wt = samples / "weights_table.json"
    if wt.is_file():
        try:
            rank_prev = ranking_index(json.loads(wt.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, TypeError, ValueError):
            rank_prev = None
    rank_names = _ranking_names(content)
    catalog_n = len(spend_catalog(content.ships, content.equip_meta))
    seen_all = _load_seen(samples)
    tu_now = catalog_n - len(seen_all)
    mass = tu_now <= 0
    phase = _apply_train_phase(cfg, kernel, nets, mass=mass)
    mode_rng = random.Random(20260817 + batch * 13)
    diag.write(
        f"session start minutes={minutes} budget_s={budget} resume_batch={batch} fresh={fresh} "
        f"infer={infer.kind}:{infer.name} gpu_fight={use_gpu} phase={phase} parallel={cfg.get('_parallel')} "
        f"tu~{max(0, tu_now)}/{catalog_n} titan_hp_max={titan_max_hp(content.titan_pvp):.0f} "
        f"pvp_loss={titan_pvp_loss(content.titan_pvp):.0f} nets={nets_dir / 'behavior.nets.pt'} "
        f"modes={cfg.get('security_modes_enabled')} lowsec_frac={cfg.get('lowsec_frac')}"
    )
    print(
        f"train unlimited={budget is None} resume_batch={batch} gpu={use_gpu} "
        f"phase={phase} parallel={cfg.get('_parallel')} tu~{max(0, tu_now)}/{catalog_n} "
        f"titan_hp={titan_max_hp(content.titan_pvp):.0f} loss/defeat={titan_pvp_loss(content.titan_pvp):.0f} "
        f"Ctrl+C 或 samples/farm.stop 在当前代结束后停；Windows 不要对 pid 发 SIGTERM",
        flush=True,
    )
    reason = "complete"
    try:
        while True:
            if consume_stop_file(samples):
                stop["flag"] = True
                print("stop requested via samples/farm.stop; will save after current batch", flush=True)
                diag.write("session stop_requested farm.stop")
            if stop["flag"]:
                reason = "interrupt"
                break
            if budget is not None and (time.time() - t0) >= budget:
                reason = "budget"
                break
            elapsed = time.time() - t0
            resume_blob = _mid_match_resume_blob(samples, batch)
            if resume_blob:
                mode = str(resume_blob.get("security_mode") or "nullsec")
            else:
                mode = _pick_security_mode(cfg, mode_rng)
            cfg["_security_mode"] = mode
            if mode == "lowsec":
                n_par = int(cfg.get("lowsec_tables") or 40)
                if resume_blob:
                    try:
                        sess = json.loads((samples / "session.json").read_text(encoding="utf-8")) or {}
                        n_par = int(sess.get("parallel") or n_par)
                    except (json.JSONDecodeError, TypeError, ValueError, OSError):
                        n_par = max(n_par, len(resume_blob.get("leagues") or []))
                cfg["_parallel"] = max(1, n_par)
                cfg["_phase"] = "lowsec"
                phase = "lowsec"
            else:
                phase = _apply_train_phase(cfg, kernel, nets, mass=mass)
                if resume_blob:
                    try:
                        sess = json.loads((samples / "session.json").read_text(encoding="utf-8")) or {}
                        n_par = int(sess.get("parallel") or 0)
                    except (json.JSONDecodeError, TypeError, ValueError, OSError):
                        n_par = 0
                    if n_par > 0:
                        cfg["_parallel"] = n_par
            print(
                f"batch {batch} start mode={mode} phase={phase} parallel={cfg.get('_parallel')} "
                f"seen={len(seen_all)}/{catalog_n} elapsed={elapsed:.0f}s 中文UTF8",
                flush=True,
            )
            diag.write(
                f"session batch={batch} start mode={mode} phase={phase} parallel={cfg.get('_parallel')} elapsed_s={elapsed:.1f}"
            )
            last = run_match(
                content=content,
                cfg=cfg,
                kernel=kernel,
                use_gpu=use_gpu,
                infer=infer,
                base=prior,
                seed=20260815 + batch * 997,
                gen_i=batch,
                diag=diag,
                replay=replay,
                replay_log=replay_log,
                nets=nets,
                resume_blob=resume_blob,
            )
            prior = last["export"]
            (samples / "behavior.genome.json").write_text(
                json.dumps(last["export"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            try:
                save_elite_generation(
                    samples, batch, last["export"], keep=int(cfg.get("elite_pool_keep") or 8)
                )
            except Exception:
                pass
            nets.save()
            seen_all |= {str(x) for x in (last.get("seen") or [])}
            _save_seen(samples, seen_all)
            add_fsm_stats(fsm_life, last.get("fsm"))
            fsm_line = format_fsm_feedback(last.get("fsm"), gen=batch, lifetime=fsm_life)
            table = _write_ranking(
                samples, content, last["export"], seen_all, security_mode=mode, fsm_line=fsm_line
            )
            tu = _tu_n(table)
            rank_now = ranking_index(table)
            print(format_ranking_delta(rank_prev, rank_now, rank_names, gen=batch), flush=True)
            print(fsm_line, flush=True)
            rank_prev = rank_now
            mass = tu <= 0
            if mode != "lowsec":
                phase = _apply_train_phase(cfg, kernel, nets, mass=mass)
            par = int(cfg.get("_parallel") or 1)
            batch += 1
            _write_session(
                samples,
                next_batch=batch,
                reason="checkpoint",
                wall_s=time.time() - t0,
                extra={
                    "phase": phase,
                    "tu": tu,
                    "catalog": catalog_n,
                    "seen": len(seen_all),
                    "parallel": par,
                    "security_mode": mode,
                    "mid_match": False,
                    "fsm_lifetime": fsm_life,
                    "fsm": last.get("fsm") or empty_fsm_stats(),
                },
            )
            elapsed = time.time() - t0
            elites = [e["seat_id"] for e in last["elites"]]
            print(
                f"batch {batch - 1} done mode={mode} phase={phase} parallel={par} tu={tu}/{catalog_n} "
                f"elapsed={elapsed:.0f}s elites={elites}",
                flush=True,
            )
            diag.write(
                f"session batch={batch - 1} done mode={mode} phase={phase} parallel={par} tu={tu} seen={len(seen_all)} "
                f"elapsed_s={elapsed:.1f} elites={elites}"
            )
    except KeyboardInterrupt:
        reason = "interrupt"
        print("KeyboardInterrupt; keeping last checkpoint", flush=True)
        diag.write("session KeyboardInterrupt")
    wall = time.time() - t0
    _write_session(
        samples,
        next_batch=batch,
        reason=reason,
        wall_s=wall,
        extra={"fsm_lifetime": fsm_life},
    )
    diag.write(f"session end reason={reason} next_batch={batch} started_at={start_batch} wall_s={wall:.1f}")
    diag.close()
    replay_log.close()
    print(f"diag {log_dir / 'diag.log'}")
    print(f"replay {log_dir / 'replay.txt'}")
    print(f"nets {nets_dir / 'behavior.nets.pt'}")
    print(f"session next_batch={batch} reason={reason} wall_s={wall:.0f}")
    if last:
        print(f"elites {[e['seat_id'] for e in last['elites']]}")


if __name__ == "__main__":
    configure_utf8_stdio()
    mins = None
    args = sys.argv[1:]
    fresh = "--fresh" in args
    if "--minutes" in args:
        i = args.index("--minutes")
        mins = float(args[i + 1])
    main(minutes=mins, fresh=fresh)
