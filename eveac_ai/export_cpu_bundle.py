"""Export FourNetPack Sequential weights to Godot CPU JSON bundles."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples" / "nets"
PT_PATH = SAMPLES / "behavior.nets.pt"
GENOME_PATH = ROOT / "samples" / "behavior.genome.json"
OUT_FARM = SAMPLES / "model_bundle"
OUT_GODOT = Path(r"H:\game_dev\eveautochess-dev\godot_project\data\ai\model_bundle")

NET_NAMES = ("titan", "match_global", "ops", "shop", "fit", "place")


def _dump_mlp(mod) -> dict:
	out: dict = {"kind": "sequential_linear_silu"}
	sd = mod.state_dict()
	for name, tensor in sd.items():
		parts = name.split(".")
		if len(parts) < 2:
			continue
		idx = parts[0]
		kind = parts[1]
		arr = tensor.detach().cpu().contiguous().numpy()
		if kind == "weight":
			out["W%s" % idx] = [float(x) for x in arr.reshape(-1).tolist()]
			out["out%s" % idx] = int(arr.shape[0])
			out["in%s" % idx] = int(arr.shape[1])
		elif kind == "bias":
			out["b%s" % idx] = [float(x) for x in arr.reshape(-1).tolist()]
	return out


def _sha256_bytes(parts: list[bytes]) -> str:
	h = hashlib.sha256()
	for p in parts:
		h.update(p)
	return h.hexdigest()


def main() -> int:
	import torch

	from eveac_ai.nets.pack import FourNetPack

	if not PT_PATH.is_file():
		print("missing", PT_PATH, file=sys.stderr)
		return 2
	device = torch.device("cpu")
	pack = FourNetPack(device=device, dir_path=SAMPLES)
	mods = {
		"titan": pack.titan,
		"match_global": pack.match_global,
		"ops": pack.ops,
		"shop": pack.shop,
		"fit": pack.fit_scorer,
		"place": pack.place,
	}
	schema_ver = "1"
	content_rev = ""
	if GENOME_PATH.is_file():
		try:
			g = json.loads(GENOME_PATH.read_text(encoding="utf-8"))
			if isinstance(g, dict):
				content_rev = str(g.get("content_rev") or g.get("schema_ver") or "")
		except (OSError, json.JSONDecodeError):
			content_rev = ""
	payloads: dict[str, dict] = {}
	hash_parts: list[bytes] = []
	for name in NET_NAMES:
		payloads[name] = _dump_mlp(mods[name])
		raw = json.dumps(payloads[name], separators=(",", ":"), ensure_ascii=True).encode("utf-8")
		hash_parts.append(name.encode("ascii") + b"\n" + raw)
	model_bundle_hash = _sha256_bytes(hash_parts)
	manifest = {
		"schema_ver": schema_ver,
		"content_rev": content_rev,
		"target": "cpu",
		"security_mode": "both",
		"model_bundle_hash": model_bundle_hash,
		"nets": list(NET_NAMES),
	}
	for dest in (OUT_FARM, OUT_GODOT):
		dest.mkdir(parents=True, exist_ok=True)
		(dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
		for name in NET_NAMES:
			(dest / ("%s.json" % name)).write_text(
				json.dumps(payloads[name], indent=2) + "\n", encoding="utf-8"
			)
		print("wrote", dest, "hash=", model_bundle_hash[:16])
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
