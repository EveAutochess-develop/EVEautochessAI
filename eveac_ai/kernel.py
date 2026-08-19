"""CPU BattleKernel: COMBAT §14 lock cadence, unmanned 3D orbit, 20 Hz."""

from __future__ import annotations

import time
from typing import Any

from eveac_ai.board_geom import clamp_xz, field_cell_xz, play_bounds_cells
from eveac_ai.content import Content
from eveac_ai.formulas import (
    combat_params,
    lock_time_s,
    missile_damage_factor,
    turret_hit_chance,
    turret_hit_quality,
)
from eveac_ai.drones import DRONE_REVIVE_DELAY_S, revive_drone, spawn_combat_unmanned, step_orbit
from eveac_ai.lance import flush_salvo, tick_lances, weapons_suppressed
from eveac_ai.rng import FarmRng
from eveac_ai.equipment_tick import apply_equips
from eveac_ai.fetters import apply_fetters
from eveac_ai.prepare import is_cyno_flagship
from eveac_ai.ship import SimShip, apply_heal, apply_hit, damage_sum, spawn_from_content


def _scale_dmg(base: dict[str, float], mul: float) -> dict[str, float]:
    return {k: v * mul for k, v in base.items()}


class BattleKernel:
    def __init__(self, content: Content) -> None:
        self.content = content
        self.cp = combat_params(content.combat)
        self.dt = float(content.match_flow.get("sim_fixed_step_s", 0.05) or 0.05)
        self.max_s = float(content.match_flow.get("battle_duration_s", 900) or 900)
        self.min_s = float(content.match_flow.get("min_battle_duration_s", 1.25) or 1.25)
        self.wall_s = float("inf")
        self.last_sim_s = 0.0

    def _event_step(
        self,
        ships: list[SimShip],
        missiles: list[dict[str, Any]],
        revive_q: list[dict[str, Any]],
        t: float,
        retarget_acc: float,
        retarget_s: float,
    ) -> float:
        """Next Δt: movers/missiles stay on the fixed step; idle / static DPS jumps to the next event."""
        dt = self.dt
        if t < self.min_s:
            return dt
        if missiles:
            return dt
        nxt = self.max_s - t
        nxt = min(nxt, max(dt, retarget_s - retarget_acc))
        moving = False
        for s in ships:
            if s.destroyed:
                continue
            if s.hold_until > t:
                nxt = min(nxt, s.hold_until - t)
                continue
            if s.is_protect_target:
                continue
            if s.is_mining_ship and not s.is_unmanned:
                continue
            if s.unmanned_kind == "mining_excavator":
                moving = True
                break
            # Unmanned orbiters keep the fixed step (3D ring motion).
            if s.is_unmanned and s.unmanned_kind.find("sentry") < 0:
                moving = True
                break
            tgt = None
            if s.lock_uid:
                for e in ships:
                    if e.uid == s.lock_uid and e.alive():
                        tgt = e
                        break
            if tgt is None:
                moving = True
                break
            dist = s.dist_to(tgt)
            if s.is_missile() and s.attack_range >= 100.0:
                moving = True
                break
            hold_cap = s.attack_range * float(self.cp.get("approach_factor") or 0.9)
            if dist > hold_cap + 0.05:
                moving = True
                break
        if moving:
            return dt
        for s in ships:
            if s.destroyed or s.is_protect_target:
                continue
            if s.is_mining_ship and not s.is_unmanned:
                continue
            if s.hold_until > t:
                continue
            cd = s.attack_duration - (t - s.last_attack)
            if cd > 0:
                nxt = min(nxt, cd)
            lk = s.lock_need - s.lock_timer
            if lk > 0:
                nxt = min(nxt, lk)
        for q in revive_q:
            due = float(q.get("revive_at") or 0.0)
            if due > t:
                nxt = min(nxt, due - t)
        if nxt <= dt:
            return dt
        return min(nxt, self.max_s - t)

    def spawn_fleet(self, ship_ids: list[str], team: int, uid0: int, x0: float, pos: list[dict[str, Any]] | None = None) -> list[SimShip]:
        ships: list[SimShip] = []
        rows = pos if pos else [{"ship_id": sid, "x": 0, "z": i} for i, sid in enumerate(ship_ids)]
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
            eqs = p.get("equips") or []
            apply_equips(s, self.content, [str(e).split(":", 1)[0] for e in eqs])
            if is_cyno_flagship(hull) or p.get("cyno_hold"):
                s.hold_until = float(self.content.match_flow.get("cyno_jump_delay_s") or 90.0)
            if p.get("pve_freighter"):
                s.is_protect_target = True
            ships.append(s)
        return ships

    def fight(
        self,
        *,
        fleet_a: list[str],
        fleet_b: list[str],
        seed: int,
        match_id: str,
        round_i: int,
        seat_a: int,
        seat_b: int,
        pos_a: list[dict[str, Any]] | None = None,
        pos_b: list[dict[str, Any]] | None = None,
        titan_a: str = "",
        titan_b: str = "",
        **_extra: Any,
    ) -> dict[str, Any]:
        rng = FarmRng(seed)
        ships = self.spawn_fleet(fleet_a, 0, 1, 0.0, pos_a) + self.spawn_fleet(fleet_b, 1, 100, 12.0, pos_b)
        nxt = max((s.uid for s in ships), default=0) + 1
        spawn_combat_unmanned(self.content, ships, nxt)
        uid_next = [max((s.uid for s in ships), default=0) + 1]
        apply_fetters(ships, self.content, {0: titan_a, 1: titan_b})
        missiles: list[dict[str, Any]] = []
        kills: list[dict[str, Any]] = []
        revive_q: list[dict[str, Any]] = []
        traj: list[dict[str, Any]] = []
        hp_trends: list[dict[str, Any]] = []
        lock_tl: list[dict[str, Any]] = []
        t = 0.0
        wall0 = time.monotonic()
        last_snap = -1.0
        snap_every = 2.0
        max_snaps = 24
        wu = self.cp["world_units_per_cell"]
        pierce = bool(self.content.combat.get("shield_overflow_pierces_armor", True))
        cap_on = bool(self.content.combat.get("capacitor_combat_enabled", True))
        gate = self.cp["cap_disable_attack_function_pct"]
        hit_r = self.cp["missile_hit_radius_wu"] / max(wu, 0.001)
        retarget_s = float(self.content.combat.get("retarget_interval_s", 10.0) or 10.0)
        search_s = float(self.content.combat.get("no_target_search_s", 0.5) or 0.5)
        approach = float(self.cp.get("approach_factor") or 0.9)
        speed_m_per_cell = float(self.content.combat.get("speed_meters_per_cell") or 750.0)
        move_scale = float(self.cp.get("move_speed_scale") or 1.65)
        bounds = play_bounds_cells(self.content.board, world_units_per_cell=wu)
        kite_cap = max(8.0, min(14.0, (bounds[1] - bounds[0] + bounds[3] - bounds[2]) * 0.35))
        retarget_acc = 0.0

        def move_cells_of(s: SimShip) -> float:
            # Godot ShipUnit.combat_move_speed → cells/s
            return max(0.5 / max(wu, 0.001), (s.speed_base / max(speed_m_per_cell, 1.0)) * move_scale)

        def living(team: int | None = None) -> list[SimShip]:
            out = [s for s in ships if s.alive()]
            if team is None:
                return out
            return [s for s in out if s.team == team]

        def manned_field(team: int) -> list[SimShip]:
            return [s for s in living(team) if not s.is_unmanned and not s.is_protect_target]

        def combat_presence(team: int) -> bool:
            return any(s.has_offense() or s.hold_until > 0 for s in manned_field(team))

        def by_uid() -> dict[int, SimShip]:
            return {s.uid: s for s in ships}

        while t < self.max_s:
            if time.monotonic() - wall0 >= float(self.wall_s):
                break
            tick = self._event_step(ships, missiles, revive_q, t, retarget_acc, retarget_s)
            t += tick
            a_manned = manned_field(0)
            b_manned = manned_field(1)
            if t >= self.min_s and (not a_manned or not b_manned):
                break
            if t >= self.min_s and a_manned and b_manned and not combat_presence(0) and not combat_presence(1):
                break
            self._cull_orphans(ships)
            self._tick_revives(ships, revive_q, uid_next, t)
            flush_salvo(ships, self.content, t)
            tick_lances(ships, self.content, tick, t, pierce, kills, revive_q)
            retarget_acc += tick
            periodic = retarget_acc >= retarget_s
            if periodic:
                retarget_acc = 0.0
            lookup = by_uid()
            for s in living():
                if s.hold_until > t:
                    continue
                if cap_on and s.cap_capacity > 0:
                    s.cap_current = min(s.cap_capacity, s.cap_current + (s.cap_capacity / s.cap_recharge_s) * tick)
                if s.unmanned_kind == "mining_excavator":
                    s.lock_uid = 0
                    step_orbit(s, 6.0, 0.0, s.z, tick, 1.35, rng.randf(), rng.randf())
                    continue
                if s.is_mining_ship and not s.is_unmanned:
                    s.lock_uid = 0
                    continue
                if s.is_protect_target:
                    s.lock_uid = 0
                    continue
                tgt = self._update_targeting(s, living, lookup, periodic, search_s, t, rng, lock_tl)
                if tgt is None:
                    continue
                s.lock_timer += tick
                s.lock_s += tick
                dist = s.dist_to(tgt)
                if s.is_unmanned and s.unmanned_kind.find("sentry") < 0:
                    rad = max(s.optimal_cells, 2.0) if s.unmanned_kind == "fighter" else max(0.9, min(s.attack_range * 0.8, 1.6))
                    step_orbit(s, tgt.x, tgt.y, tgt.z, tick, rad, rng.randf(), rng.randf())
                    dist = s.dist_to(tgt)
                else:
                    if s.is_missile() and s.attack_range >= 100.0:
                        # Godot: non-sleeper missiles kite within playable bounds.
                        desired = max(self.cp["min_engagement_cells"], min(kite_cap, dist + 1.0))
                        if dist < desired - 0.05:
                            step = min(move_cells_of(s) * tick, desired - dist)
                            if dist > 1e-6:
                                s.x -= (tgt.x - s.x) / dist * step
                                s.z -= (tgt.z - s.z) / dist * step
                        elif dist > desired + 0.05:
                            step = min(move_cells_of(s) * tick, dist - desired)
                            if dist > 1e-6:
                                s.x += (tgt.x - s.x) / dist * step
                                s.z += (tgt.z - s.z) / dist * step
                    else:
                        hold_cap = s.attack_range * approach
                        desired = max(self.cp["min_engagement_cells"], hold_cap)
                        if dist > hold_cap + 0.05:
                            step = min(move_cells_of(s) * tick, dist - desired)
                            if dist > 1e-6:
                                s.x += (tgt.x - s.x) / dist * step
                                s.z += (tgt.z - s.z) / dist * step
                                if s.chase_y:
                                    s.y += (tgt.y - s.y) / dist * step
                    s.x, s.z = clamp_xz(s.x, s.z, bounds)
                if (not s.is_logistic) and (s.repair.get("shield") or s.repair.get("armor") or s.repair.get("structure")):
                    if t - s.last_repair >= max(s.attack_duration, 1.0):
                        s.last_repair = t
                        apply_heal(s, s.repair.get("shield") or 0.0, s.repair.get("armor") or 0.0, s.repair.get("structure") or 0.0)
                if s.is_logistic:
                    ally = self._best_hurt_ally(s, living(s.team))
                    if ally and t - s.last_attack >= s.attack_duration and s.lock_timer >= s.lock_need:
                        if (not cap_on) or s.cap_frac() >= gate:
                            s.last_attack = t
                            self._spend_cap(s)
                            got = apply_heal(ally, s.repair["shield"], s.repair["armor"], s.repair["structure"])
                            s.dmg_out += got
                    continue
                if not s.has_offense():
                    continue
                if weapons_suppressed(s):
                    continue
                if t - s.last_attack < s.attack_duration:
                    continue
                if cap_on and s.cap_frac() < gate:
                    continue
                if s.lock_timer < s.lock_need:
                    continue
                dist = s.dist_to(tgt)
                if dist > s.attack_range + 0.001:
                    continue
                s.last_attack = t
                self._spend_cap(s)
                if s.is_missile():
                    fac = missile_damage_factor(
                        tgt.signature,
                        tgt.speed,
                        s.explosion_radius,
                        s.explosion_velocity,
                        s.missile_drf,
                        s.missile_drs,
                        missile_drs_default=self.cp["missile_drs_default"],
                    )
                    missiles.append(
                        {
                            "x": s.x,
                            "y": s.y,
                            "z": s.z,
                            "src": s.uid,
                            "tgt": tgt.uid,
                            "dmg": _scale_dmg(s.damage, fac),
                            "spd": self.cp["missile_speed_cells_per_s"],
                        }
                    )
                else:
                    p = turret_hit_chance(
                        s.tracking,
                        s.optimal_cells,
                        s.falloff_cells,
                        s.optimal_sig,
                        tgt.speed,
                        tgt.signature,
                        dist,
                        meters_per_cell=self.cp["meters_per_cell"],
                        tracking_meters_per_cell=self.cp["tracking_meters_per_cell"],
                        hit_chance_min=self.cp["hit_chance_min"],
                        hit_chance_max=self.cp["hit_chance_max"],
                    )
                    q = turret_hit_quality(rng.randf(), p)
                    if q > 0.0:
                        res = apply_hit(tgt, _scale_dmg(s.damage, q), pierce=pierce)
                        s.dmg_out += float(res["dealt"])
                        if res["destroyed"]:
                            kills.append({"t": round(t, 3), "killer": s.uid, "victim": tgt.uid, "killer_ship_id": s.ship_id, "ship_id": tgt.ship_id, "victim_unmanned": bool(tgt.is_unmanned)})
                            lock_tl.append({"t": round(t, 3), "uid": s.uid, "tgt": tgt.uid, "event": "kill"})
                            self._schedule_revive(tgt, t, revive_q)
            self._tick_missiles(missiles, ships, tick, hit_r, pierce, t, kills, revive_q)
            if any(s.is_protect_target and not s.alive() for s in ships):
                break
            if t - last_snap >= snap_every and len(traj) < max_snaps:
                last_snap = t
                traj.append(
                    {
                        "t": round(t, 3),
                        "pos": [{"uid": s.uid, "x": round(s.x, 3), "z": round(s.z, 3)} for s in ships if s.alive()],
                    }
                )
                hp_trends.append(
                    {
                        "t": round(t, 3),
                        "hp": [
                            {
                                "uid": s.uid,
                                "s": round(s.shield, 1),
                                "a": round(s.armor, 1),
                                "h": round(s.structure, 1),
                            }
                            for s in ships
                        ],
                    }
                )

        self.last_sim_s = float(t)
        a_manned = manned_field(0)
        b_manned = manned_field(1)
        # region agent log
        if t < 2.0:
            try:
                import json
                from pathlib import Path

                rec = {
                    "sessionId": "519d6e",
                    "timestamp": int(time.time() * 1000),
                    "hypothesisId": "B",
                    "location": "kernel.py:fight:end",
                    "message": "cpu fight ended under 2s",
                    "data": {
                        "t": round(t, 4),
                        "min_s": self.min_s,
                        "n_ships": len(ships),
                        "a_manned": len(a_manned),
                        "b_manned": len(b_manned),
                        "a_off": int(combat_presence(0)),
                        "b_off": int(combat_presence(1)),
                    },
                }
                with Path(r"H:\debug-519d6e.log").open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            except Exception:
                pass
        # endregion
        a_win = bool(a_manned) and not b_manned
        b_win = bool(b_manned) and not a_manned
        # Godot: both manned alive or empty mutual → draw (no HP crown).

        def pack(team: int, seat: int, won: bool) -> dict[str, Any]:
            members = [s for s in ships if s.team == team]
            hp_frac = 0.0
            tot = 0.0
            for s in members:
                if s.is_unmanned:
                    continue
                tot += s.max_structure
                hp_frac += max(0.0, s.structure)
            titan_hp = 80.0 if won else 60.0
            if tot > 0:
                titan_hp = 40.0 + 60.0 * (hp_frac / tot) if won else 20.0 + 40.0 * (hp_frac / tot)
            return {
                "seat_id": seat,
                "won": won,
                "titan_hp": round(titan_hp, 2),
                "loss_comp_gold": 0 if won else 1,
                "rank_hint": sum(s.dmg_out for s in members),
                "ships": [
                    {
                        "uid": str(s.uid),
                        "ship_id": s.ship_id,
                        "dmg_out": round(s.dmg_out, 2),
                        "dmg_in": round(s.dmg_in, 2),
                        "repaired": round(s.repaired, 2),
                        "cap_used": round(s.cap_used, 2),
                        "lock_s": round(s.lock_s, 2),
                        "survived": s.alive(),
                        "is_unmanned": s.is_unmanned,
                        "unmanned_kind": s.unmanned_kind,
                        "star": s.star,
                        "mother_uid": s.mother_uid,
                    }
                    for s in members
                ],
                "kill_calendar": [k for k in kills if any(s.uid == k["killer"] and s.team == team for s in ships)],
                "traj_snapshots": traj if team == 0 else [],
                "hp_trends": hp_trends if team == 0 else [],
                "lock_timeline": [e for e in lock_tl if e.get("event") != "lock_start"][:40],
            }

        return {
            "schema_ver": "1",
            "backend": "cpu",
            "match_id": match_id,
            "round": round_i,
            "sim_s": float(t),
            "end_reason": (
                "wipe"
                if a_win or b_win
                else (
                    "cap"
                    if t >= self.max_s - 0.15
                    else ("empty" if (not a_manned and not b_manned) else ("wall" if t < self.max_s - 0.15 and time.monotonic() - wall0 >= float(self.wall_s) else "no_off"))
                )
            ),
            "seats": [pack(0, seat_a, a_win), pack(1, seat_b, b_win)],
        }

    def _schedule_revive(self, victim: SimShip, t: float, q: list[dict[str, Any]]) -> None:
        if not victim.is_unmanned or victim.mother_uid <= 0:
            return
        q.append(
            {
                "mother_uid": victim.mother_uid,
                "drone_id": victim.ship_id,
                "revive_at": t + DRONE_REVIVE_DELAY_S,
                "star": max(1, int(victim.star)),
            }
        )

    def _tick_revives(self, ships: list[SimShip], q: list[dict[str, Any]], uid_next: list[int], t: float) -> None:
        if not q:
            return
        by = {s.uid: s for s in ships}
        left: list[dict[str, Any]] = []
        for e in q:
            mom = by.get(int(e["mother_uid"]))
            if mom is None or not mom.alive():
                continue
            if t < float(e["revive_at"]):
                left.append(e)
                continue
            have = sum(1 for s in ships if s.alive() and s.is_unmanned and s.mother_uid == mom.uid)
            d = revive_drone(self.content, mom, int(e["drone_id"]), uid_next[0], int(e["star"]), have)
            uid_next[0] += 1
            ships.append(d)
            by[d.uid] = d
        q[:] = left

    def _cull_orphans(self, ships: list[SimShip]) -> None:
        by = {s.uid: s for s in ships}
        for s in ships:
            if not s.is_unmanned or not s.alive() or s.mother_uid <= 0:
                continue
            mom = by.get(s.mother_uid)
            if mom is None or not mom.alive():
                s.destroyed = True
                s.structure = 0.0

    def _find_target(self, s: SimShip, enemies: list[SimShip]) -> SimShip | None:
        cand = list(enemies)
        if s.skip_unmanned:
            cand = [e for e in cand if not e.is_unmanned]
        if not cand:
            return None
        return min(cand, key=lambda e: (s.dist_to(e), e.structure / max(e.max_structure, 1.0), e.uid))

    def _assign_lock(self, s: SimShip, tgt: SimShip | None, t: float, lock_tl: list[dict[str, Any]]) -> None:
        uid = tgt.uid if tgt else 0
        if s.lock_uid != uid:
            s.lock_uid = uid
            s.lock_timer = 0.0
            if tgt:
                s.lock_need = lock_time_s(s.scan, tgt.signature, self.cp["lock_time_constant"])
                lock_tl.append({"t": round(t, 3), "uid": s.uid, "tgt": tgt.uid, "event": "lock_start"})
            s.stall_uid = uid
            s.stall_until = t + 30.0
            if tgt:
                s.stall_sh, s.stall_ar, s.stall_st = tgt.shield, tgt.armor, tgt.structure

    def _hp_stall(self, s: SimShip, tgt: SimShip, living, t: float, rng: FarmRng, lock_tl: list) -> bool:
        if s.is_logistic or tgt is None:
            return False
        if s.stall_uid != tgt.uid:
            s.stall_uid = tgt.uid
            s.stall_until = t + 30.0 + rng.randf() * 10.0
            s.stall_sh, s.stall_ar, s.stall_st = tgt.shield, tgt.armor, tgt.structure
            return False
        if tgt.shield < s.stall_sh - 0.01 or tgt.armor < s.stall_ar - 0.01 or tgt.structure < s.stall_st - 0.01:
            s.stall_until = t + 30.0 + rng.randf() * 10.0
            s.stall_sh, s.stall_ar, s.stall_st = tgt.shield, tgt.armor, tgt.structure
            return False
        if t < s.stall_until:
            return False
        allies = [a for a in living(s.team) if a.uid != s.uid and a.alive() and a.lock_uid and a.lock_uid != tgt.uid]
        alt = None
        if allies:
            focus = {a.lock_uid for a in allies}
            enemies = living(1 - s.team)
            opts = [e for e in enemies if e.uid in focus]
            if opts:
                alt = self._find_target(s, opts) or opts[0]
        if alt is None:
            alt = self._find_target(s, living(1 - s.team))
        if alt is None or alt.uid == tgt.uid:
            s.stall_until = t + 30.0
            return False
        self._assign_lock(s, alt, t, lock_tl)
        s.no_target_acc = 0.0
        return True

    def _update_targeting(
        self,
        s: SimShip,
        living,
        lookup: dict[int, SimShip],
        periodic: bool,
        search_s: float,
        t: float,
        rng: FarmRng,
        lock_tl: list[dict[str, Any]],
    ) -> SimShip | None:
        if s.is_unmanned and s.mother_uid and s.unmanned_kind != "heavy_repair_drone":
            mom = lookup.get(s.mother_uid)
            if mom and mom.alive() and mom.has_offense():
                mt = lookup.get(mom.lock_uid)
                if mt and mt.alive():
                    self._assign_lock(s, mt, t, lock_tl)
                    return mt
        tgt = lookup.get(s.lock_uid) if s.lock_uid else None
        if tgt is not None and not tgt.alive():
            tgt = None
        if tgt and not s.is_logistic and self._hp_stall(s, tgt, living, t, rng, lock_tl):
            return lookup.get(s.lock_uid)
        need = tgt is None
        if not need and not s.is_logistic and periodic:
            need = True
        if not need:
            s.no_target_acc = 0.0
            return tgt
        if tgt is not None and not tgt.alive():
            nxt = self._find_target(s, living(1 - s.team))
            self._assign_lock(s, nxt, t, lock_tl)
            s.no_target_acc = 0.0
            return nxt
        if tgt is not None and tgt.alive():
            nxt = self._find_target(s, living(1 - s.team))
            self._assign_lock(s, nxt, t, lock_tl)
            return nxt
        s.no_target_acc += self.dt
        if s.no_target_acc >= search_s:
            s.no_target_acc = 0.0
            nxt = self._find_target(s, living(1 - s.team))
            self._assign_lock(s, nxt, t, lock_tl)
            return nxt
        return None

    def _spend_cap(self, s: SimShip) -> None:
        if s.cap_capacity <= 0:
            return
        cost = s.cap_cost
        if cost < 0:
            cost = s.cap_capacity * self.cp["cap_drain_fraction_per_cycle"]
        s.cap_current = max(0.0, s.cap_current - cost)
        s.cap_used += cost

    def _best_hurt_ally(self, logi: SimShip, allies: list[SimShip]) -> SimShip | None:
        hurt = [a for a in allies if a.uid != logi.uid and a.alive() and (a.shield < a.max_shield or a.armor < a.max_armor or a.structure < a.max_structure)]
        if not hurt:
            return None
        return min(hurt, key=lambda a: (a.structure / max(a.max_structure, 1.0), logi.dist_to(a)))

    def _tick_missiles(
        self,
        missiles: list[dict[str, Any]],
        ships: list[SimShip],
        dt: float,
        hit_r: float,
        pierce: bool,
        t: float,
        kills: list[dict[str, Any]],
        revive_q: list[dict[str, Any]] | None = None,
    ) -> None:
        by_uid = {s.uid: s for s in ships}
        keep: list[dict[str, Any]] = []
        for m in missiles:
            tgt = by_uid.get(int(m["tgt"]))
            src = by_uid.get(int(m["src"]))
            if tgt is None or not tgt.alive():
                continue
            dx = tgt.x - float(m["x"])
            dy = tgt.y - float(m.get("y") or 0.0)
            dz = tgt.z - float(m["z"])
            dist = (dx * dx + dy * dy + dz * dz) ** 0.5
            step = float(m["spd"]) * dt
            if dist <= hit_r or dist <= step:
                res = apply_hit(tgt, m["dmg"], pierce=pierce)
                if src:
                    src.dmg_out += float(res["dealt"])
                if res["destroyed"]:
                    kills.append({
                        "t": round(t, 3),
                        "killer": int(m["src"]),
                        "victim": tgt.uid,
                        "killer_ship_id": src.ship_id if src else "",
                        "ship_id": tgt.ship_id,
                        "victim_unmanned": bool(tgt.is_unmanned),
                    })
                    if revive_q is not None:
                        self._schedule_revive(tgt, t, revive_q)
                continue
            m["x"] = float(m["x"]) + dx / dist * step
            m["y"] = float(m.get("y") or 0.0) + dy / dist * step
            m["z"] = float(m["z"]) + dz / dist * step
            keep.append(m)
        missiles[:] = keep
