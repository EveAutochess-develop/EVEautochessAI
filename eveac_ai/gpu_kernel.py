"""Batched GPU (or torch-CPU) battle inference. Missiles resolve instantly with DR."""

from __future__ import annotations

from typing import Any

from eveac_ai.board_geom import field_cell_xz, play_bounds_cells
from eveac_ai.content import Content
from eveac_ai.formulas import combat_params
from eveac_ai.equipment_tick import apply_equips
from eveac_ai.fetters import apply_fetters
from eveac_ai.drones import spawn_combat_unmanned
from eveac_ai.kernel import BattleKernel
from eveac_ai.prepare import is_cyno_flagship
from eveac_ai.ship import SimShip, spawn_from_content


class GpuBattleKernel:
    def __init__(self, content: Content, torch_device: object, *, sparse_dt: bool | None = None) -> None:
        self.content = content
        self.device = torch_device
        self.cpu = BattleKernel(content)
        self.cp = combat_params(content.combat)
        self.dt = float(content.match_flow.get("sim_fixed_step_s", 0.05) or 0.05)
        self.max_s = float(content.match_flow.get("battle_duration_s", 900) or 900)
        self.min_s = float(content.match_flow.get("min_battle_duration_s", 1.25) or 1.25)
        self.last_sim_s = 0.0
        self.last_timing: dict = {}
        # Throughput switch only — not a Godot mechanism gold standard.
        if sparse_dt is None:
            sparse_dt = bool(content.match_flow.get("gpu_sparse_dt", True))
        self.sparse_dt = bool(sparse_dt)

    def fight(self, **kwargs: Any) -> dict[str, Any]:
        return self.fight_batch([kwargs])[0]

    def fight_batch(self, jobs: list[dict[str, Any]], *, slots: int | None = None) -> list[dict[str, Any]]:
        if not jobs:
            return []
        try:
            import torch
        except ImportError:
            return [self.cpu.fight(**j) for j in jobs]
        return self._fight_batch_torch(jobs, torch, slot_n=int(slots or 0))

    def _spawn(self, fleet: list[str], team: int, uid0: int, x0: float, pos: list[dict[str, Any]] | None = None) -> list[SimShip]:
        rows = pos if pos else [{"ship_id": sid, "x": 0, "z": i} for i, sid in enumerate(fleet)]
        out: list[SimShip] = []
        wu = self.cp["world_units_per_cell"]
        board = self.content.board
        for i, p in enumerate(rows):
            hull = self.content.ships.get(str(p["ship_id"])) or {}
            cx = int(p.get("x") or 0)
            cz = int(p.get("z") or 0)
            if pos is not None:
                sx, sz = field_cell_xz(board, team, cx, cz, world_units_per_cell=wu)
            else:
                sx, sz = x0 + float(cx), float(cz) * 1.2
            s = spawn_from_content(
                self.content,
                str(p["ship_id"]),
                team,
                uid0 + i,
                star=max(1, int(p.get("star") or 1)),
                x=sx,
                z=sz,
            )
            apply_equips(s, self.content, [str(e).split(":", 1)[0] for e in (p.get("equips") or [])])
            if is_cyno_flagship(hull) or p.get("cyno_hold"):
                s.hold_until = float(self.content.match_flow.get("cyno_jump_delay_s") or 90.0)
            if p.get("pve_freighter"):
                s.is_protect_target = True
            out.append(s)
        return out

    def _fight_batch_torch(self, jobs: list[dict[str, Any]], torch: Any, *, slot_n: int = 0) -> list[dict[str, Any]]:
        import time as _time

        device = self.device
        n_jobs = len(jobs)
        sides: list[list[SimShip]] = []
        max_s = 1
        for j in jobs:
            a = self._spawn(j["fleet_a"], 0, 1, 0.0, j.get("pos_a"))
            b = self._spawn(j["fleet_b"], 1, 100, 12.0, j.get("pos_b"))
            both = a + b
            nxt = max((s.uid for s in both), default=0) + 1
            spawn_combat_unmanned(self.content, both, nxt)
            apply_fetters(both, self.content, {0: str(j.get("titan_a") or ""), 1: str(j.get("titan_b") or "")})
            sides.append(both)
            max_s = max(max_s, len(both))
        S = max_s
        slotted = int(slot_n or 0) > 0 and int(slot_n) < n_jobs
        B = int(slot_n) if slotted else n_jobs
        base_dt = self.dt
        # Few ships (early PVE / salvage): larger Δt → fewer wall ticks for 900s sim.
        # DPS averages the same; RNG sequence differs from 0.05 but farm grades stay usable.
        # Not mechanism-gold vs Godot; toggle via gpu_sparse_dt / GpuBattleKernel.sparse_dt.
        sparse = bool(self.sparse_dt) and S <= 24 and not slotted
        dt = 0.25 if sparse else base_dt
        cp = self.cp
        wu = cp["world_units_per_cell"]
        move_cells = (cp["move_speed"] * cp["move_speed_scale"]) / max(wu, 0.001)
        pierce = bool(self.content.combat.get("shield_overflow_pierces_armor", True))
        cap_on = bool(self.content.combat.get("capacitor_combat_enabled", True))
        gate = cp["cap_disable_attack_function_pct"]
        min_eng = cp["min_engagement_cells"]
        approach = float(cp.get("approach_factor") or 0.9)
        hit_r = cp["missile_hit_radius_wu"] / max(wu, 0.001)
        mis_spd = cp["missile_speed_cells_per_s"]
        retarget_s = float(self.content.combat.get("retarget_interval_s", 10.0) or 10.0)
        bounds = play_bounds_cells(self.content.board, world_units_per_cell=wu)
        kite_cap = max(8.0, min(14.0, (bounds[1] - bounds[0] + bounds[3] - bounds[2]) * 0.35))
        # One Philox stream; per-table seed mixing via offset (avoids B Python RNG kernels/tick).
        rng_seed = (int(jobs[0].get("seed") or 1) ^ (B * 10007) ^ (S * 9176)) & 0x7FFFFFFF
        rng_gen = torch.Generator(device=device)
        rng_gen.manual_seed(rng_seed if rng_seed else 1)

        def z(shape, fill=0.0, dtype=None):
            t = torch.full(shape, fill, device=device, dtype=dtype or torch.float32)
            return t

        mask = z((B, S), 0.0)
        team = z((B, S), 0.0)
        x = z((B, S), 0.0)
        yy = z((B, S), 0.0)
        zz = z((B, S), 0.0)
        sh = z((B, S), 0.0)
        ar = z((B, S), 0.0)
        st = z((B, S), 0.0)
        msh = z((B, S), 1.0)
        mar = z((B, S), 1.0)
        mst = z((B, S), 1.0)
        sh_r = z((B, S, 4), 0.0)
        ar_r = z((B, S, 4), 0.0)
        su_r = z((B, S, 4), 0.0)
        dmg = z((B, S, 4), 0.0)
        tracking = z((B, S), 0.08)
        optimal = z((B, S), 8.0)
        falloff = z((B, S), 4.0)
        opt_sig = z((B, S), 40.0)
        signature = z((B, S), 40.0)
        scan = z((B, S), 400.0)
        speed = z((B, S), 300.0)
        atk_range = z((B, S), 6.0)
        atk_dur = z((B, S), 1.0)
        cap_cap = z((B, S), 0.0)
        cap_re = z((B, S), 1.0)
        cap_cur = z((B, S), 0.0)
        cap_cost = z((B, S), 0.0)
        last_atk = z((B, S), -999.0)
        lock_t = z((B, S), 0.0)
        lock_need = z((B, S), 0.0)
        lock_idx = torch.zeros((B, S), device=device, dtype=torch.long)
        is_mis = z((B, S), 0.0)
        is_log = z((B, S), 0.0)
        has_off = z((B, S), 0.0)
        expl_r = z((B, S), 1.0)
        expl_v = z((B, S), 1.0)
        drf = z((B, S), 0.0)
        dmg_out = z((B, S), 0.0)
        dmg_in = z((B, S), 0.0)
        cap_used = z((B, S), 0.0)
        lock_s = z((B, S), 0.0)
        hold = z((B, S), 0.0)
        is_unm = z((B, S), 0.0)
        skip_unm = z((B, S), 0.0)
        skip_prot = z((B, S), 0.0)
        excav = z((B, S), 0.0)
        mining_m = z((B, S), 0.0)
        inherit = z((B, S), 0.0)
        mother_idx = torch.full((B, S), -1, device=device, dtype=torch.long)
        lock_inited = z((B, S), 0.0)
        mis_live = z((B, S), 0.0)
        mx = z((B, S), 0.0)
        my = z((B, S), 0.0)
        mz = z((B, S), 0.0)
        mdmg = z((B, S, 4), 0.0)
        mtgt = torch.zeros((B, S), device=device, dtype=torch.long)
        revive_at = z((B, S), -1.0)
        hit_r = cp["missile_hit_radius_wu"] / max(wu, 0.001)
        mis_spd = cp["missile_speed_cells_per_s"]
        retarget_s = float(self.content.combat.get("retarget_interval_s", 10.0) or 10.0)
        ship_ids: list[list[str]] = [[""] * S for _ in range(B)]
        extra_meta: list[list[dict[str, Any]]] = [[{} for _ in range(S)] for _ in range(B)]
        uids = torch.zeros((B, S), device=device, dtype=torch.long)
        alive = z((B, S), 0.0)

        keys = ("emp", "thermal", "kinetic", "explosive")

        def load_slot(bi: int, job_i: int) -> None:
            # Zero the live tensors (names rebound in-loop); do not snapshot refs at def time.
            for ten in (
                mask, team, x, yy, zz, sh, ar, st, msh, mar, mst, tracking, optimal, falloff, opt_sig, signature,
                scan, speed, atk_range, atk_dur, cap_cap, cap_re, cap_cur, cap_cost, last_atk, lock_t, lock_need,
                is_mis, is_log, has_off, expl_r, expl_v, drf, dmg_out, dmg_in, cap_used, lock_s, hold, is_unm,
                skip_unm, skip_prot, excav, mining_m, inherit, lock_inited, mis_live, mx, my, mz, revive_at, alive,
                sh_r, ar_r, su_r, dmg, mdmg, lock_idx, mother_idx, mtgt, uids,
            ):
                ten[bi].zero_()
            last_atk[bi].fill_(-999.0)
            mother_idx[bi].fill_(-1)
            revive_at[bi].fill_(-1.0)
            msh[bi].fill_(1.0)
            mar[bi].fill_(1.0)
            mst[bi].fill_(1.0)
            ship_ids[bi] = [""] * S
            extra_meta[bi] = [{} for _ in range(S)]
            ships = sides[job_i]
            for si, s in enumerate(ships):
                mask[bi, si] = 1.0
                team[bi, si] = float(s.team)
                x[bi, si] = s.x
                yy[bi, si] = s.y
                zz[bi, si] = s.z
                sh[bi, si] = s.shield
                ar[bi, si] = s.armor
                st[bi, si] = s.structure
                msh[bi, si] = s.max_shield
                mar[bi, si] = s.max_armor
                mst[bi, si] = s.max_structure
                for k, name in enumerate(keys):
                    sh_r[bi, si, k] = s.shield_resist.get(name, 0.0)
                    ar_r[bi, si, k] = s.armor_resist.get(name, 0.0)
                    su_r[bi, si, k] = s.structure_resist.get(name, 0.0)
                    dmg[bi, si, k] = s.damage.get(name, 0.0)
                tracking[bi, si] = s.tracking
                optimal[bi, si] = s.optimal_cells
                falloff[bi, si] = s.falloff_cells
                opt_sig[bi, si] = s.optimal_sig
                signature[bi, si] = s.signature
                scan[bi, si] = s.scan
                speed[bi, si] = s.speed
                atk_range[bi, si] = s.attack_range
                atk_dur[bi, si] = s.attack_duration
                cap_cap[bi, si] = s.cap_capacity
                cap_re[bi, si] = s.cap_recharge_s
                cap_cur[bi, si] = s.cap_current
                cost = s.cap_cost
                if cost < 0:
                    cost = s.cap_capacity * cp["cap_drain_fraction_per_cycle"]
                cap_cost[bi, si] = cost
                is_mis[bi, si] = 1.0 if s.is_missile() else 0.0
                is_log[bi, si] = 1.0 if s.is_logistic else 0.0
                has_off[bi, si] = 1.0 if s.has_offense() else 0.0
                expl_r[bi, si] = max(s.explosion_radius, 1.0)
                expl_v[bi, si] = max(s.explosion_velocity, 1.0)
                drf[bi, si] = s.missile_drf
                ship_ids[bi][si] = s.ship_id
                uids[bi, si] = s.uid
                hold[bi, si] = float(getattr(s, "hold_until", 0.0) or 0.0)
                is_unm[bi, si] = 1.0 if s.is_unmanned else 0.0
                skip_unm[bi, si] = 1.0 if s.skip_unmanned else 0.0
                skip_prot[bi, si] = 1.0 if s.is_protect_target else 0.0
                excav[bi, si] = 1.0 if s.unmanned_kind == "mining_excavator" else 0.0
                mining_m[bi, si] = 1.0 if (s.is_mining_ship and not s.is_unmanned) else 0.0
                inherit[bi, si] = 1.0 if (s.is_unmanned and s.unmanned_kind not in ("heavy_repair_drone", "mining_excavator") and s.mother_uid) else 0.0
                extra_meta[bi][si] = {
                    "is_unmanned": s.is_unmanned,
                    "is_protect_target": s.is_protect_target,
                    "unmanned_kind": s.unmanned_kind,
                    "star": s.star,
                    "mother_uid": s.mother_uid,
                    "uid": s.uid,
                }
            uid_map = {int(s.uid): i for i, s in enumerate(ships)}
            for si, s in enumerate(ships):
                if s.mother_uid:
                    mother_idx[bi, si] = uid_map.get(int(s.mother_uid), -1)
            alive[bi].copy_(mask[bi])

        for bi in range(B):
            load_slot(bi, bi)
        slot_job = list(range(B))
        queue = list(range(B, n_jobs)) if slotted else []
        occupied = torch.ones((B,), device=device, dtype=torch.bool)
        t_row = torch.zeros((B,), device=device)
        retarget_acc = torch.zeros((B,), device=device)
        tick_i = 0
        end_every = max(1, int(round(2.0 / max(dt, 1e-6))))  # ~2s sim between wipe checks
        end_t = torch.full((B,), -1.0, device=device)
        live_snap: dict[str, int] = {}
        kill_lists: list[list[dict[str, Any]]] = [[] for _ in range(B)]
        n_refill = 0
        n_decided = n_draw_cap = n_draw_no_off = n_draw_empty = 0
        results: list[dict[str, Any] | None] = [None] * n_jobs
        t_fight0 = _time.perf_counter()
        slot_t0 = [0.0] * B
        occupy_s = 0.0
        finishes: list[tuple[float, float]] = []

        def record_kills(atk_mask, atk_tgt, died_mask) -> None:
            if not bool(died_mask.any()):
                return
            am = atk_mask.detach().cpu()
            tg = atk_tgt.detach().cpu()
            du = is_unm.detach().cpu()
            for bi in range(B):
                bag = kill_lists[bi]
                if len(bag) >= 32:
                    continue
                for si in range(S):
                    if float(am[bi, si].item()) < 0.5:
                        continue
                    vi = int(tg[bi, si].item())
                    if vi < 0 or vi >= S:
                        continue
                    bag.append(
                        {
                            "t": round(float(t_row[bi].item()), 3),
                            "killer": int(uids[bi, si].item()) if int(uids[bi, si].item()) else si,
                            "victim": int(uids[bi, vi].item()) if int(uids[bi, vi].item()) else vi,
                            "killer_ship_id": ship_ids[bi][si],
                            "ship_id": ship_ids[bi][vi],
                            "victim_unmanned": bool(float(du[bi, vi].item()) > 0.5),
                        }
                    )
                    if len(bag) >= 32:
                        break

        def apply_hit(tgt_idx, incoming, src_idx):
            nonlocal sh, ar, st, alive, dmg_out, dmg_in
            # incoming [B,S,4] aligned to attacker; scatter to target via tgt_idx
            remain = incoming
            dealt_acc = torch.zeros((B, S), device=device)
            for hp, resist in ((sh, sh_r), (ar, ar_r), (st, su_r)):
                tgt_hp = torch.gather(hp, 1, tgt_idx.clamp(min=0))
                tgt_res = resist.gather(1, tgt_idx.clamp(min=0).unsqueeze(-1).expand(-1, -1, 4))
                raw = (remain * (1.0 - tgt_res)).sum(-1)
                take = torch.minimum(tgt_hp, raw.clamp(min=0.0))
                frac = torch.where(raw > 1e-6, take / raw.clamp(min=1e-6), torch.zeros_like(raw))
                hp_new = tgt_hp - take
                hp.scatter_(1, tgt_idx.clamp(min=0), torch.clamp(hp_new, min=0.0))
                dealt_acc = dealt_acc + take
                if pierce:
                    remain = remain * (1.0 - frac).unsqueeze(-1)
                else:
                    remain = remain * 0.0
            dmg_in.add_(dealt_acc)
            dmg_out.add_(dealt_acc)
            dead = (sh <= 0) & (ar <= 0) & (st <= 0)
            alive.mul_((~dead).to(dtype=alive.dtype)).mul_(mask)

        def full_dist_e():
            dx = x.unsqueeze(2) - x.unsqueeze(1)
            dy = yy.unsqueeze(2) - yy.unsqueeze(1)
            dz = zz.unsqueeze(2) - zz.unsqueeze(1)
            dist = torch.sqrt(dx * dx + dy * dy + dz * dz + 1e-8)
            enemy = (team.unsqueeze(2) != team.unsqueeze(1)).to(dtype=dist.dtype)
            valid = alive.unsqueeze(1) * alive.unsqueeze(2) * enemy * mask.unsqueeze(1) * mask.unsqueeze(2)
            skip_mask = skip_unm.unsqueeze(2) * is_unm.unsqueeze(1)
            return dist + (1.0 - valid) * 1e6 + skip_mask * 1e6

        def lock_tgt_dist(tgt_idx):
            tx = x.gather(1, tgt_idx.clamp(min=0))
            ty = yy.gather(1, tgt_idx.clamp(min=0))
            tz = zz.gather(1, tgt_idx.clamp(min=0))
            dx = tx - x
            dy = ty - yy
            dz = tz - zz
            return torch.sqrt(dx * dx + dy * dy + dz * dz + 1e-8)

        def pack_one(bi: int, job: dict[str, Any], t_end: float) -> dict[str, Any]:
            nonlocal n_decided, n_draw_cap, n_draw_no_off, n_draw_empty
            al = alive[bi].detach().cpu()
            stc = st[bi].detach().cpu()
            dout = dmg_out[bi].detach().cpu()
            din = dmg_in[bi].detach().cpu()
            cu = cap_used[bi].detach().cpu()
            ls = lock_s[bi].detach().cpu()
            msk = mask[bi].detach().cpu()
            tm = team[bi].detach().cpu()

            def pack(team_id: int) -> tuple[list, bool]:
                members = []
                for si in range(S):
                    if msk[si].item() < 0.5:
                        continue
                    if int(tm[si].item()) != team_id:
                        continue
                    surv = al[si].item() > 0.5 and stc[si].item() > 0
                    meta = extra_meta[bi][si] or {}
                    members.append(
                        {
                            "uid": str(meta.get("uid", si + (1 if team_id == 0 else 100))),
                            "ship_id": ship_ids[bi][si],
                            "dmg_out": round(float(dout[si].item()), 2),
                            "dmg_in": round(float(din[si].item()), 2),
                            "repaired": 0.0,
                            "cap_used": round(float(cu[si].item()), 2),
                            "lock_s": round(float(ls[si].item()), 2),
                            "survived": bool(surv),
                            "is_unmanned": bool(meta.get("is_unmanned")),
                            "is_protect_target": bool(meta.get("is_protect_target")),
                            "unmanned_kind": str(meta.get("unmanned_kind") or ""),
                            "star": int(meta.get("star") or 1),
                            "mother_uid": int(meta.get("mother_uid") or 0),
                        }
                    )
                manned_live = any(
                    (not m.get("is_unmanned")) and (not m.get("is_protect_target")) and m.get("survived") for m in members
                )
                return members, manned_live

            ma, a_live = pack(0)
            mb, b_live = pack(1)
            a_win = a_live and not b_live
            b_win = b_live and not a_live
            if a_win or b_win:
                end_reason = "wipe"
                n_decided += 1
            elif t_end >= self.max_s - 0.15:
                end_reason = "cap"
                n_draw_cap += 1
            elif (not a_live) and (not b_live):
                end_reason = "empty"
                n_draw_empty += 1
            else:
                end_reason = "no_off"
                n_draw_no_off += 1

            def seat_row(members, seat, won):
                manned = [m for m in members if not m.get("is_unmanned")]
                live_n = sum(1 for m in manned if m.get("survived"))
                n = max(len(manned), 1)
                titan_hp = (40.0 + 60.0 * (live_n / n)) if won else (20.0 + 40.0 * (live_n / n))
                return {
                    "seat_id": seat,
                    "won": won,
                    "titan_hp": round(titan_hp, 2),
                    "loss_comp_gold": 0 if won else 1,
                    "rank_hint": sum(m["dmg_out"] for m in members),
                    "ships": members,
                    "kill_calendar": list(kill_lists[bi]),
                    "traj_snapshots": [],
                    "hp_trends": [],
                    "lock_timeline": [],
                }

            return {
                "schema_ver": "1",
                "backend": "gpu",
                "device": str(device),
                "match_id": job.get("match_id", ""),
                "round": job.get("round_i", 0),
                "sim_s": t_end,
                "end_reason": end_reason,
                "seats": [
                    seat_row(ma, job["seat_a"], a_win),
                    seat_row(mb, job["seat_b"], b_win),
                ],
            }

        def emit_finished(just_ended) -> None:
            nonlocal n_refill, occupy_s
            idx = just_ended.nonzero(as_tuple=False).view(-1).detach().cpu().tolist()
            now = _time.perf_counter() - t_fight0
            for bi in idx:
                bi = int(bi)
                ji = slot_job[bi]
                if results[ji] is None:
                    sim_done = float(end_t[bi].item())
                    results[ji] = pack_one(bi, jobs[ji], sim_done)
                    occupy_s += now - float(slot_t0[bi])
                    finishes.append((now, sim_done))
                if slotted and queue:
                    nj = queue.pop(0)
                    load_slot(bi, nj)
                    slot_job[bi] = nj
                    t_row[bi] = 0.0
                    end_t[bi] = -1.0
                    retarget_acc[bi] = 0.0
                    kill_lists[bi] = []
                    occupied[bi] = True
                    n_refill += 1
                    slot_t0[bi] = _time.perf_counter() - t_fight0
                elif slotted:
                    occupied[bi] = False
                    mask[bi].zero_()
                    alive[bi].zero_()

        while bool(occupied.any()):
            t_row = t_row + occupied.to(dtype=t_row.dtype) * dt
            tick_i += 1
            t_b = t_row.unsqueeze(1)
            lock_alive = alive.gather(1, lock_idx.clamp(min=0))
            retarget_acc = retarget_acc + occupied.to(dtype=retarget_acc.dtype) * dt
            periodic = retarget_acc >= retarget_s
            retarget_acc = torch.where(periodic, torch.zeros_like(retarget_acc), retarget_acc)
            dead_lock = lock_alive < 0.5
            need_periodic = ((alive > 0) & (is_log < 0.5)) | (lock_inited < 0.5) | dead_lock
            need_sticky = (lock_inited < 0.5) | dead_lock
            need = torch.where(periodic.unsqueeze(1), need_periodic, need_sticky)
            need = need & (excav < 0.5) & (mining_m < 0.5) & (skip_prot < 0.5)
            # Full S×S: always when sparse (S small); else only on retarget demand.
            if sparse or bool(periodic.any()) or bool(need.any()):
                dist_e = full_dist_e()
                closest = dist_e.argmin(dim=-1)
                lock_idx = torch.where(need.to(dtype=torch.bool), closest, lock_idx)
            lock_inited = torch.where(need.to(dtype=torch.bool), torch.ones_like(lock_inited), lock_inited)
            mom_ok = (mother_idx >= 0) & (inherit > 0.5)
            safe_m = mother_idx.clamp(min=0)
            m_alive = alive.gather(1, safe_m)
            m_off = has_off.gather(1, safe_m)
            m_lock = lock_idx.gather(1, safe_m)
            take_inh = mom_ok & (m_alive > 0.5) & (m_off > 0.5) & (lock_inited.gather(1, safe_m) > 0.5)
            lock_idx = torch.where(take_inh, m_lock, lock_idx)
            orphan = (is_unm > 0.5) & (mother_idx >= 0) & (m_alive < 0.5)
            alive.mul_((~orphan).to(dtype=alive.dtype))
            due = (revive_at >= 0) & (t_b >= revive_at)
            can_rev = due & (mother_idx >= 0) & (m_alive > 0.5)
            drop_rev = due & ~can_rev
            sh = torch.where(can_rev, msh, sh)
            ar = torch.where(can_rev, mar, ar)
            st = torch.where(can_rev, mst, st)
            alive.copy_(torch.where(can_rev, mask, alive))
            revive_at = torch.where(can_rev | drop_rev, torch.full_like(revive_at, -1.0), revive_at)
            tgt = lock_idx
            tgt_dist = lock_tgt_dist(tgt)
            tgt_sig = signature.gather(1, tgt.clamp(min=0))
            changed = need.to(dtype=torch.bool)
            k = cp["lock_time_constant"]
            need_t = k / (scan.clamp(min=1.0) * tgt_sig.clamp(min=1.0))
            lock_need = torch.where(changed, need_t, lock_need)
            lock_t = torch.where(changed, torch.zeros_like(lock_t), lock_t + dt)
            lock_s = lock_s + dt * alive

            if cap_on:
                cap_cur = torch.minimum(cap_cap, cap_cur + (cap_cap / cap_re.clamp(min=1e-3)) * dt)

            hold_cap = atk_range * approach
            missile_kite = (is_mis > 0.5) & (atk_range >= 100.0)
            kite_desired = torch.maximum(
                torch.full_like(tgt_dist, min_eng),
                torch.minimum(torch.full_like(tgt_dist, kite_cap), tgt_dist + 1.0),
            )
            desired = torch.where(missile_kite, kite_desired, torch.maximum(torch.full_like(tgt_dist, min_eng), hold_cap))
            step_in = torch.minimum(torch.full_like(tgt_dist, move_cells * dt), (tgt_dist - desired).clamp(min=0.0))
            step_out = torch.minimum(torch.full_like(tgt_dist, move_cells * dt), (desired - tgt_dist).clamp(min=0.0))
            tx = x.gather(1, tgt)
            tz = zz.gather(1, tgt)
            nx_in = torch.where(tgt_dist > 1e-6, (tx - x) / tgt_dist * step_in, torch.zeros_like(x))
            nz_in = torch.where(tgt_dist > 1e-6, (tz - zz) / tgt_dist * step_in, torch.zeros_like(zz))
            nx_out = torch.where(tgt_dist > 1e-6, (x - tx) / tgt_dist * step_out, torch.zeros_like(x))
            nz_out = torch.where(tgt_dist > 1e-6, (zz - tz) / tgt_dist * step_out, torch.zeros_like(zz))
            can_move = (alive > 0) & (t_b >= hold) & (excav < 0.5) & (mining_m < 0.5) & (skip_prot < 0.5)
            halt = ((skip_prot > 0.5) & (alive < 0.5) & (mask > 0.5)).sum(-1, keepdim=True) > 0
            can_move = can_move & (~halt)
            far = can_move & (~missile_kite) & (tgt_dist > hold_cap + 0.05)
            kite_out = can_move & missile_kite & (tgt_dist < desired - 0.05)
            kite_in = can_move & missile_kite & (tgt_dist > desired + 0.05)
            x = x + nx_in * (far | kite_in).to(x.dtype) + nx_out * kite_out.to(x.dtype)
            zz = zz + nz_in * (far | kite_in).to(zz.dtype) + nz_out * kite_out.to(zz.dtype)
            x = x.clamp(bounds[0], bounds[1])
            zz = zz.clamp(bounds[2], bounds[3])

            can = (
                (alive > 0)
                & (has_off > 0)
                & (is_log < 0.5)
                & (excav < 0.5)
                & (mining_m < 0.5)
                & (skip_prot < 0.5)
                & ((t_b - last_atk) >= atk_dur)
                & (lock_t >= lock_need)
                & (tgt_dist <= atk_range + 0.001)
                & (t_b >= hold)
                & (~halt)
            )
            if cap_on:
                frac = torch.where(cap_cap > 0, cap_cur / cap_cap.clamp(min=1e-6), torch.ones_like(cap_cur))
                can = can & (frac >= gate)
            can_gun = can & (is_mis < 0.5)
            can_mis = can & (is_mis > 0.5) & (mis_live < 0.5)
            last_atk = torch.where(can_gun | can_mis, t_b.expand_as(last_atk), last_atk)
            cap_cur = torch.where(can_gun | can_mis, (cap_cur - cap_cost).clamp(min=0.0), cap_cur)
            cap_used = cap_used + cap_cost * (can_gun | can_mis).to(cap_used.dtype)

            tgt_spd = speed.gather(1, tgt)
            tgt_sig = signature.gather(1, tgt)
            d_track = tgt_dist * cp["tracking_meters_per_cell"]
            d_m = tgt_dist * cp["meters_per_cell"]
            omega = tgt_spd / d_track.clamp(min=1.0)
            tracking_term = (omega / tracking.clamp(min=1e-6)) * (opt_sig.clamp(min=1.0) / tgt_sig.clamp(min=1.0))
            r_opt = optimal * cp["meters_per_cell"]
            r_fo = (falloff * cp["meters_per_cell"]).clamp(min=1.0)
            range_term = ((d_m - r_opt).clamp(min=0.0)) / r_fo
            p = (0.5 ** (tracking_term * tracking_term + range_term * range_term)).clamp(cp["hit_chance_min"], cp["hit_chance_max"])
            xrnd = torch.rand((B, S), device=device, generator=rng_gen)
            q = torch.where(xrnd > p, torch.zeros_like(p), torch.where(xrnd < 0.01, torch.full_like(p, 3.0), xrnd + 0.5))
            sig_ratio = tgt_sig / expl_r.clamp(min=1.0)
            vt_eff = tgt_spd.clamp(min=0.0001)
            drs_eff = torch.where(drf > 1.0, torch.full_like(drf, cp["missile_drs_default"]), torch.ones_like(drf))
            drf_exp = torch.where(
                drf <= 0.0,
                torch.zeros_like(drf),
                torch.where(drf > 1.0, torch.log(drf.clamp(min=1e-6)) / torch.log(drs_eff.clamp(min=1.0001)), drf),
            )
            speed_term = (sig_ratio * (expl_v / vt_eff)).clamp(min=0.0) ** drf_exp.clamp(min=0.0)
            mis_fac = torch.where(drf <= 0.0, torch.minimum(torch.ones_like(sig_ratio), sig_ratio), torch.minimum(torch.ones_like(sig_ratio), torch.minimum(sig_ratio, speed_term)))
            scale = torch.where(is_mis > 0.5, mis_fac, q)
            incoming = dmg * scale.unsqueeze(-1) * can_gun.to(dmg.dtype).unsqueeze(-1)
            was = alive.clone()
            apply_hit(tgt, incoming, None)
            died = (was > 0.5) & (alive < 0.5)
            record_kills(can_gun & died.gather(1, tgt.clamp(min=0)), tgt, died)
            died_u = died & (is_unm > 0.5) & (mother_idx >= 0)
            revive_at = torch.where(died_u, t_b + 400.0, revive_at)

            mis_live = torch.where(can_mis, torch.ones_like(mis_live), mis_live)
            mx = torch.where(can_mis, x, mx)
            my = torch.where(can_mis, yy, my)
            mz = torch.where(can_mis, zz, mz)
            mtgt = torch.where(can_mis, tgt, mtgt)
            mdmg = torch.where(can_mis.unsqueeze(-1), dmg * mis_fac.unsqueeze(-1), mdmg)
            fly = mis_live > 0.5
            txm = x.gather(1, mtgt.clamp(min=0))
            tym = yy.gather(1, mtgt.clamp(min=0))
            tzm = zz.gather(1, mtgt.clamp(min=0))
            t_al = alive.gather(1, mtgt.clamp(min=0))
            mdx = txm - mx
            mdy = tym - my
            mdz = tzm - mz
            md = torch.sqrt(mdx * mdx + mdy * mdy + mdz * mdz + 1e-8)
            step_m = torch.full_like(md, mis_spd * dt)
            hit_m = fly & ((t_al < 0.5) | (md <= hit_r) | (md <= step_m))
            land = fly & (t_al > 0.5) & ((md <= hit_r) | (md <= step_m))
            was_m = alive.clone()
            apply_hit(mtgt, mdmg * land.to(mdmg.dtype).unsqueeze(-1), None)
            died_m = (was_m > 0.5) & (alive < 0.5)
            record_kills(land & died_m.gather(1, mtgt.clamp(min=0)), mtgt, died_m)
            died2 = died_m & (is_unm > 0.5) & (mother_idx >= 0)
            revive_at = torch.where(died2, t_b + 400.0, revive_at)
            nxm = torch.where(md > 1e-6, mdx / md * step_m, torch.zeros_like(mx))
            mx = mx + nxm * (fly & ~hit_m).to(mx.dtype)
            my = my + torch.where(md > 1e-6, mdy / md * step_m, torch.zeros_like(my)) * (fly & ~hit_m).to(my.dtype)
            mz = mz + torch.where(md > 1e-6, mdz / md * step_m, torch.zeros_like(mz)) * (fly & ~hit_m).to(mz.dtype)
            mis_live = mis_live * (~hit_m).to(mis_live.dtype)

            if tick_i % end_every == 0:
                halted = halt.squeeze(-1)
                manned = alive * (is_unm < 0.5) * (skip_prot < 0.5)
                a_n = (manned * (team < 0.5)).sum(-1)
                b_n = (manned * (team > 0.5)).sum(-1)
                a_off = ((manned * (team < 0.5) * has_off).sum(-1) > 0) | ((manned * (team < 0.5) * (hold > 0)).sum(-1) > 0)
                b_off = ((manned * (team > 0.5) * has_off).sum(-1) > 0) | ((manned * (team > 0.5) * (hold > 0)).sum(-1) > 0)
                wiped = (a_n < 0.5) | (b_n < 0.5)
                no_off = (a_n > 0.5) & (b_n > 0.5) & (~a_off) & (~b_off)
                late = t_row >= self.min_s
                capped = occupied & (t_row >= self.max_s)
                live_fights = occupied & (end_t < 0) & ~(halted | (late & (wiped | no_off)) | capped)
                just_ended = occupied & (end_t < 0) & ~live_fights
                end_t = torch.where(just_ended, t_row, end_t)
                emit_finished(just_ended)
                running = occupied & (end_t < 0)
                live_n = int(running.sum().item())
                clock = float(t_row[occupied].max().item()) if bool(occupied.any()) else 0.0
                for key, th in (("t2", 2.0), ("t10", 10.0), ("t60", 60.0), ("t300", 300.0)):
                    if key not in live_snap and clock >= th:
                        live_snap[key] = live_n
                if not bool(running.any()) and not queue:
                    break
                # Static DPS stretch: nobody needs to close range / kite → jump toward next volleys.
                if sparse and not slotted and bool((t_row >= self.min_s).any()):
                    movers = (far | kite_in | kite_out) & (~halt)
                    flying = (mis_live > 0.5) & (~halt)
                    if (not bool(movers.any())) and (not bool(flying.any())):
                        wait_atk = (atk_dur - (t_b - last_atk)).clamp(min=0.0)
                        wait_lock = (lock_need - lock_t).clamp(min=0.0)
                        wait = torch.where(alive > 0.5, torch.minimum(wait_atk, wait_lock), torch.full_like(wait_atk, 1e9))
                        wait = torch.where((has_off > 0.5) & (skip_prot < 0.5) & (excav < 0.5) & (mining_m < 0.5), wait, torch.full_like(wait, 1e9))
                        jump_cap = (retarget_s - retarget_acc).clamp(min=0.0)
                        jump = float(wait.min().clamp(min=0.0).item())
                        jump = min(jump, float(jump_cap.min().item()) if B else 0.0)
                        if jump > dt * 1.5:
                            t_row = torch.minimum(t_row + (jump - dt), torch.full_like(t_row, self.max_s))
                            lock_t = lock_t + (jump - dt)
                            lock_s = lock_s + (jump - dt) * alive
                            retarget_acc = torch.minimum(torch.full_like(retarget_acc, retarget_s), retarget_acc + (jump - dt))
                            if cap_on:
                                cap_cur = torch.minimum(cap_cap, cap_cur + (cap_cap / cap_re.clamp(min=1e-3)) * (jump - dt))

        leftover = occupied & (end_t < 0)
        end_t = torch.where(leftover, t_row, end_t)
        emit_finished(leftover)
        packs = []
        for p in results:
            if p is None:
                raise RuntimeError("gpu fight_batch slot missed a job pack")
            packs.append(p)
        et = [float(p.get("sim_s") or 0.0) for p in packs]
        n = max(1, len(et))
        wall_s = _time.perf_counter() - t_fight0
        if not slotted:
            occupy_s = float(B) * wall_s
        self.last_sim_s = float(max(et) if et else 0.0)
        self.last_timing = {
            "sparse": bool(sparse),
            "dt": float(dt),
            "S_kernel": int(S),
            "B": int(B),
            "n_jobs": int(n_jobs),
            "slots": int(B),
            "slotted": bool(slotted),
            "n_refill": int(n_refill),
            "end_every": int(end_every),
            "ticks": int(tick_i),
            "batch_t": round(self.last_sim_s, 3),
            "wall_s": round(wall_s, 3),
            "occupy_s": round(occupy_s, 3),
            "finishes": [(round(float(w), 3), round(float(s), 3)) for w, s in finishes],
            "n_lt_2": sum(1 for v in et if v < 2.0),
            "n_lt_min": sum(1 for v in et if v < self.min_s),
            "n_lt_10": sum(1 for v in et if v < 10.0),
            "n_lt_60": sum(1 for v in et if v < 60.0),
            "n_lt_300": sum(1 for v in et if v < 300.0),
            "n_cap": sum(1 for v in et if v >= self.max_s - 0.15),
            "end_min": round(min(et), 3) if et else None,
            "end_max": round(max(et), 3) if et else None,
            "end_mean": round(sum(et) / n, 3),
            "live_snap": live_snap,
            "sparse_gate_S24": int(S) <= 24,
            "n_decided": n_decided,
            "n_draw_cap": n_draw_cap,
            "n_draw_no_off": n_draw_no_off,
            "n_draw_empty": n_draw_empty,
        }
        # region agent log
        try:
            import json
            import time
            from pathlib import Path

            rec = {
                "sessionId": "519d6e",
                "timestamp": int(time.time() * 1000),
                "hypothesisId": "A",
                "runId": "slot-refill",
                "location": "gpu_kernel.py:fight_batch:end",
                "message": "slotted fight_batch packs",
                "data": {
                    "slotted": bool(slotted),
                    "slots": int(B),
                    "n_jobs": int(n_jobs),
                    "n_refill": int(n_refill),
                    "n_packs": len(packs),
                    "last_sim_s": round(self.last_sim_s, 3),
                    "end_mean": self.last_timing.get("end_mean"),
                    "n_cap": self.last_timing.get("n_cap"),
                    "sparse": bool(sparse),
                    "dt": float(dt),
                    "ticks": int(tick_i),
                },
            }
            with Path(r"H:\debug-519d6e.log").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            pass
        # endregion
        return packs
