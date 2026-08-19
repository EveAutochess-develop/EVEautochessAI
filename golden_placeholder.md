# 金标

权威：[`AI_SELFPLAY.md`](../EVEautochessAI-design/docs/AI_SELFPLAY.md) §10。

## 已做到（本里程碑）

- **公式**：`tests/test_formulas.py` 对照 `CombatFormulas`（命中、品质、锁定时间、导弹因子）
- **打层**：`tests/test_apply_hit.py` 盾击穿甲
- **核回归**：`tests/golden/1v1.json` 同快照同种子 → 同胜负、同击毁顺序（Python `FarmRng`，**不是** Godot PCG）
- **TelemetryPack**：逐舰输出/承受/存活、击毁日历；轨迹/血量趋势降频（约 2s，最多 24 拍）

`backend`：`gpu`（CUDA 批桌，导弹即时结算，不与 CPU 逐 tick 同胜负）/ `cpu` / `cpu_stub` / `gpu_stub`。

## 未做到

同快照 + 同 `MatchRng` 种子与游戏 `combat_resolver.gd` **同胜负**（需 Godot 无头导出 + PCG 对齐）。无人、功能桶、混合长枪、诱导不在本核。
