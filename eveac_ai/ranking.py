"""Gold-spend ranking: T0–T48 + TU. Ships and equips mixed on one ladder."""

from __future__ import annotations

from typing import Any

SCHEMA_VER = "1"
N_STRENGTH = 49  # T0 .. T48
TU = "TU"
TIER_ORDER = [f"T{i}" for i in range(N_STRENGTH)] + [TU]


def is_titan_hull(hull: dict[str, Any] | None) -> bool:
    hull = hull or {}
    if str(hull.get("capital_role", "")).lower() == "titan":
        return True
    if str(hull.get("ship_group", "")).lower() == "titan":
        return True
    return False


def is_gold_ship(hull: dict[str, Any] | None) -> bool:
    hull = hull or {}
    if is_titan_hull(hull):
        return False
    if hull.get("shop_eligible") is False:
        return False
    if float(hull.get("cost") or 0.0) <= 0:
        return False
    return True


def is_gold_equip(meta: dict[str, Any] | None) -> bool:
    meta = meta or {}
    if meta.get("shop_pool") is False:
        return False
    if meta.get("implant"):
        return False
    if float(meta.get("cost") or 0.0) <= 0:
        return False
    return True


def item_key(kind: str, iid: str) -> str:
    return f"{kind}:{iid}"


def parse_item_key(key: str) -> tuple[str, str]:
    kind, _, iid = str(key).partition(":")
    if kind in ("ship", "equip") and iid:
        return kind, iid
    return "ship", str(key)


def rankable_ship_ids(ships: dict[str, dict[str, Any]]) -> list[str]:
    return [sid for sid, hull in ships.items() if is_gold_ship(hull)]


def rankable_equip_ids(equip_meta: dict[str, dict[str, Any]]) -> list[str]:
    return [eid for eid, meta in equip_meta.items() if is_gold_equip(meta)]


def spend_catalog(
    ships: dict[str, dict[str, Any]],
    equip_meta: dict[str, dict[str, Any]],
) -> list[tuple[str, str]]:
    out = [("ship", sid) for sid in rankable_ship_ids(ships)]
    out.extend(("equip", eid) for eid in rankable_equip_ids(equip_meta))
    return out


def quantize_ships(
    genome: dict[str, Any],
    all_ship_ids: list[str],
    *,
    seen_ids: set[str] | None = None,
    content_rev: str = "",
    ship_names: dict[str, str] | None = None,
    hulls: dict[str, dict[str, Any]] | None = None,
    equip_meta: dict[str, dict[str, Any]] | None = None,
    equip_names: dict[str, str] | None = None,
) -> dict[str, Any]:
    """One mixed T0–T48+TU ladder. seen_ids may be raw ship ids or kind:id keys."""
    ships_map = hulls or {}
    eq_map = equip_meta or {}
    if ships_map or eq_map:
        catalog = spend_catalog(ships_map, eq_map)
    else:
        catalog = [("ship", str(s)) for s in all_ship_ids]

    titans = genome.get("titan_ids") or list((genome.get("titan_slices") or {}).keys())
    slices = genome.get("titan_slices") or {}
    names = dict(ship_names or {})
    names.update(equip_names or {})

    raw_seen = {str(s) for s in (seen_ids or [])}
    seen_keys: set[str] = set()
    for s in raw_seen:
        if s.startswith("ship:") or s.startswith("equip:"):
            seen_keys.add(s)
        else:
            seen_keys.add(item_key("ship", s))
            seen_keys.add(item_key("equip", s))

    def mean_w(kind: str, iid: str) -> float:
        field = "ship" if kind == "ship" else "equip"
        vals = []
        for t in titans:
            sl = slices.get(t) or {}
            vals.append(float((sl.get(field) or {}).get(iid, 0.5)))
        return sum(vals) / max(1, len(vals))

    all_keys = [item_key(k, i) for k, i in catalog]
    known = [(k, i) for k, i in catalog if item_key(k, i) in seen_keys]
    unknown = [(k, i) for k, i in catalog if item_key(k, i) not in seen_keys]
    # Same weight → same tier (unlimited items). Distinct weights → distinct
    # T0..T48 when ≤49 levels; if more unique weights, merge by weight-rank
    # spacing (never by item count). TU = unused only; no internal order.
    by_w: dict[float, list[tuple[str, str]]] = {}
    for kind, iid in known:
        w = mean_w(kind, iid)
        by_w.setdefault(w, []).append((kind, iid))
    unique_ws = sorted(by_w.keys(), reverse=True)
    buckets: dict[str, list[tuple[str, str]]] = {t: [] for t in TIER_ORDER}
    n_u = len(unique_ws)
    for ui, w in enumerate(unique_ws):
        if n_u <= N_STRENGTH:
            tier_i = ui
        else:
            tier_i = int(round(ui * (N_STRENGTH - 1) / (n_u - 1)))
        items = sorted(by_w[w], key=lambda ki: (ki[0], ki[1]))
        buckets[f"T{tier_i}"].extend(items)
    buckets[TU] = sorted(unknown, key=lambda ki: (ki[0], ki[1]))

    listed = []
    for t in TIER_ORDER:
        listed.extend(item_key(k, i) for k, i in buckets[t])
    if set(listed) != set(all_keys) or len(listed) != len(all_keys):
        raise RuntimeError("ranking must list every gold-spend item exactly once")

    tiers: dict[str, dict[str, Any]] = {}
    for t in TIER_ORDER:
        items = [{"kind": k, "id": i} for k, i in buckets[t]]
        tiers[t] = {
            "items": items,
            "ships": [i for k, i in buckets[t] if k == "ship"],
            "equips": [i for k, i in buckets[t] if k == "equip"],
        }
    return {
        "schema_ver": SCHEMA_VER,
        "content_rev": content_rev or genome.get("content_rev", ""),
        "kind": "relative_tiers",
        "note": "T0 important … T48 unimportant; same tier=equal, unlimited items; TU=unused (no internal rank). Ships+equips mixed. Titans excluded.",
        "tier_order": list(TIER_ORDER),
        "tiers": tiers,
        "item_names": {item_key(k, i): names.get(i, i) for k, i in catalog},
        "ship_names": {sid: names.get(sid, sid) for _, sid in catalog if _ == "ship"} if names else {},
    }


def format_ranking_table(table: dict[str, Any], ship_names: dict[str, str] | None = None) -> str:
    names = dict(table.get("item_names") or {})
    names.update(ship_names or table.get("ship_names") or {})
    lines = [
        "黄币花费排行  T0重要 → T48不重要  |  TU=未用过（无内部名次）  |  同档并列且不限数量  |  舰装混排  |  不含泰坦",
        "档   | 花费项（同档=同等）",
        "-----+------------------------------",
    ]
    for key in table.get("tier_order") or TIER_ORDER:
        items = ((table.get("tiers") or {}).get(key) or {}).get("items") or []
        if not items:
            cell = "（空）"
        else:
            bits = []
            for it in items:
                kind = it.get("kind", "ship")
                iid = str(it.get("id", ""))
                tag = "舰" if kind == "ship" else "装"
                label = names.get(item_key(kind, iid), names.get(iid, iid))
                bits.append(f"{tag}:{iid} {label}".strip())
            cell = " · ".join(bits)
        lines.append(f"{key:<4} | {cell}")
    lines.append("")
    return "\n".join(lines)


def ranking_index(table: dict[str, Any]) -> dict[str, str]:
    """item_key → tier name only (T0..T48 or TU). No within-tier / within-TU order."""
    out: dict[str, str] = {}
    for tname in table.get("tier_order") or TIER_ORDER:
        items = ((table.get("tiers") or {}).get(tname) or {}).get("items") or []
        for it in items:
            kind = str(it.get("kind") or "ship")
            iid = str(it.get("id") or "")
            out[item_key(kind, iid)] = str(tname)
    return out


def format_ranking_delta(
    old: dict[str, str] | None,
    new: dict[str, str],
    names: dict[str, str],
    *,
    gen: int,
) -> str:
    lines: list[str] = []
    if not old:
        return f"排行 Δ gen={gen}  （首代建档，无对照）"
    for key, nv in new.items():
        ov = old.get(key)
        if ov == nv:
            continue
        kind, _, iid = key.partition(":")
        tag = "舰" if kind == "ship" else "装"
        label = names.get(key, names.get(iid, iid))
        if ov is None:
            lines.append(f"排行 Δ gen={gen}  {tag}:{iid} {label}  TU → {nv}")
        else:
            lines.append(f"排行 Δ gen={gen}  {tag}:{iid} {label}  {ov} → {nv}")
    if not lines:
        return f"排行 Δ gen={gen}  （无档位变化）"
    return "\n".join(lines)
