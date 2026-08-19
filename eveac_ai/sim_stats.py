"""Wall-clock binned sim throughput. Slot refill makes wave sim/x misleading."""

from __future__ import annotations

from typing import Any

DEFAULT_INTERVAL_S = 10.0


def bucket_finishes(
    finishes: list[tuple[float, float]],
    interval_s: float,
    *,
    cap_s: float = 899.0,
) -> list[dict[str, Any]]:
    """finishes: (wall_offset_s, sim_s) in completion order."""
    interval_s = float(interval_s)
    if interval_s <= 0 or not finishes:
        return []
    tmax = max(float(w) for w, _ in finishes)
    nbin = max(1, int(tmax / interval_s) + 1)
    bins = [{"n": 0, "sim_sum": 0.0, "n_cap": 0} for _ in range(nbin)]
    last = nbin - 1
    for w, sim in finishes:
        i = int(float(w) / interval_s)
        if i < 0:
            i = 0
        if i > last:
            i = last
        bins[i]["n"] += 1
        bins[i]["sim_sum"] += float(sim)
        if float(sim) >= cap_s:
            bins[i]["n_cap"] += 1
    return bins


def summarize_finishes(
    finishes: list[tuple[float, float]],
    *,
    wall_s: float,
    slots: int,
    occupy_s: float,
    interval_s: float = DEFAULT_INTERVAL_S,
    cap_s: float = 899.0,
) -> dict[str, Any]:
    n = len(finishes)
    wall = max(float(wall_s), 1e-6)
    k = max(int(slots), 1)
    sim_sum = sum(float(s) for _, s in finishes)
    n_cap = sum(1 for _, s in finishes if float(s) >= cap_s)
    bins = bucket_finishes(finishes, interval_s, cap_s=cap_s)
    util = float(occupy_s) / (k * wall)
    if util < 0:
        util = 0.0
    return {
        "wall_s": round(float(wall_s), 3),
        "slots": k,
        "n_jobs": n,
        "rate": n / wall if n else 0.0,
        "util": util,
        "mean_sim": (sim_sum / n) if n else 0.0,
        "sim_sum": sim_sum,
        "n_cap": n_cap,
        "occupy_s": float(occupy_s),
        "interval_s": float(interval_s),
        "bins": [int(b["n"]) for b in bins],
    }


def summary_from_timing(timing: dict[str, Any] | None, *, interval_s: float = DEFAULT_INTERVAL_S) -> dict[str, Any]:
    timing = timing or {}
    finishes: list[tuple[float, float]] = []
    for row in timing.get("finishes") or []:
        if isinstance(row, (list, tuple)) and len(row) >= 2:
            finishes.append((float(row[0]), float(row[1])))
    return summarize_finishes(
        finishes,
        wall_s=float(timing.get("wall_s") or 0.0),
        slots=int(timing.get("slots") or timing.get("B") or 1),
        occupy_s=float(timing.get("occupy_s") or 0.0),
        interval_s=float(interval_s),
        cap_s=899.0,
    )


def add_summaries(dst: dict[str, Any], src: dict[str, Any] | None) -> dict[str, Any]:
    if not src:
        return dst
    if not dst:
        dst.update(
            {
                "wall_s": 0.0,
                "slots": int(src.get("slots") or 1),
                "n_jobs": 0,
                "sim_sum": 0.0,
                "n_cap": 0,
                "occupy_s": 0.0,
                "interval_s": float(src.get("interval_s") or DEFAULT_INTERVAL_S),
            }
        )
    dst["wall_s"] = float(dst.get("wall_s") or 0.0) + float(src.get("wall_s") or 0.0)
    dst["n_jobs"] = int(dst.get("n_jobs") or 0) + int(src.get("n_jobs") or 0)
    dst["sim_sum"] = float(dst.get("sim_sum") or 0.0) + float(src.get("sim_sum") or 0.0)
    dst["n_cap"] = int(dst.get("n_cap") or 0) + int(src.get("n_cap") or 0)
    dst["occupy_s"] = float(dst.get("occupy_s") or 0.0) + float(src.get("occupy_s") or 0.0)
    dst["slots"] = max(int(dst.get("slots") or 1), int(src.get("slots") or 1))
    return dst


def format_round_stats(summary: dict[str, Any] | None) -> str:
    if not summary or int(summary.get("n_jobs") or 0) <= 0:
        return ""
    interval = int(round(float(summary.get("interval_s") or DEFAULT_INTERVAL_S)))
    bins = summary.get("bins") or []
    bin_s = ",".join(str(int(x)) for x in bins) if bins else "-"
    return (
        f"wall={float(summary['wall_s']):.1f}s slots={int(summary['slots'])} "
        f"rate={float(summary['rate']):.2f}/s util={float(summary['util']):.2f} "
        f"mean_sim={float(summary['mean_sim']):.0f} "
        f"cap={int(summary['n_cap'])}/{int(summary['n_jobs'])} "
        f"bin{interval}s={bin_s}"
    )


def format_gen_stats(agg: dict[str, Any] | None, *, gen: int) -> str:
    if not agg or int(agg.get("n_jobs") or 0) <= 0:
        return f"sim.gen={gen} (no fights)"
    n = int(agg["n_jobs"])
    wall = max(float(agg.get("wall_s") or 0.0), 1e-6)
    slots = max(int(agg.get("slots") or 1), 1)
    occupy = float(agg.get("occupy_s") or 0.0)
    mean_sim = float(agg.get("sim_sum") or 0.0) / n
    util = occupy / (slots * wall) if occupy else 0.0
    return (
        f"sim.gen={gen} wall={wall:.0f}s slots={slots} rate={n / wall:.2f}/s "
        f"util={util:.2f} mean_sim={mean_sim:.0f} cap={int(agg.get('n_cap') or 0)}/{n}"
    )
