"""20-generation self-play. Ranking excludes titans."""

from __future__ import annotations

import json
from pathlib import Path

from eveac_ai.board_view import ship_label
from eveac_ai.capped_log import CappedLog
from eveac_ai.content import Content, load_config
from eveac_ai.device import resolve_device
from eveac_ai.gpu_kernel import GpuBattleKernel
from eveac_ai.kernel import BattleKernel
from eveac_ai.match20 import ROOT, run_match
from eveac_ai.priors import load_bootstrap
from eveac_ai.ranking import format_ranking_table, quantize_ships, spend_catalog
from eveac_ai.replay import ReplayWriter


def write_ranking(samples: Path, content: Content, export: dict, seen: set[str]) -> None:
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
    samples.mkdir(parents=True, exist_ok=True)
    (samples / "behavior.genome.json").write_text(json.dumps(export, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (samples / "weights_table.json").write_text(json.dumps(table, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (samples / "ranking_ships.txt").write_text(format_ranking_table(table, names), encoding="utf-8")
    tu_n = len(((table.get("tiers") or {}).get("TU") or {}).get("items") or [])
    return tu_n, len(catalog)


def main() -> None:
    cfg = load_config()
    n_match = int(cfg.get("train_matches", 20))
    content = Content(cfg=cfg)
    base = load_bootstrap(content=content)
    infer = resolve_device("auto")
    use_gpu = infer.kind == "cuda" and infer.torch_device is not None
    kernel = GpuBattleKernel(content, infer.torch_device) if use_gpu else BattleKernel(content)
    log_dir = ROOT / "samples" / "logs"
    samples = ROOT / cfg.get("out_dir", "samples")
    diag = CappedLog(log_dir / "diag.log", kind="diag")
    replay_log = CappedLog(log_dir / "replay.txt", kind="replay")
    replay = ReplayWriter(replay_log, content, content.board)
    seen: set[str] = set()
    diag.write(f"train start matches={n_match} infer={infer.kind}:{infer.name} spend_items={len(spend_catalog(content.ships, content.equip_meta))}")
    for g in range(n_match):
        out = run_match(
            content=content,
            cfg=cfg,
            kernel=kernel,
            use_gpu=use_gpu,
            infer=infer,
            base=base,
            seed=20260815 + g * 997,
            gen_i=g,
            diag=diag,
            replay=replay,
            replay_log=replay_log,
        )
        seen |= {str(x) for x in out["seen"]}
        base = out["export"]
        tu_n, n_all = write_ranking(samples, content, base, seen)
        print(f"gen {g+1}/{n_match} elites={[e['seat_id'] for e in out['elites']]} seen={len(seen)} TU={tu_n}/{n_all}")
        diag.write(f"train gen={g} elites={[e['seat_id'] for e in out['elites']]} seen={len(seen)} tu={tu_n}")
    diag.close()
    replay_log.close()
    print(f"done ranking {samples / 'ranking_ships.txt'}")


if __name__ == "__main__":
    main()
