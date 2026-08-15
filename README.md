# EVEautochessAI · 无头互训农场

独立进程：世代编排、拟真 stub、导出 `behavior.genome` + T0–Tn 排行。  
**不进** Godot 工程。

- GitHub：https://github.com/EveAutochess-develop/EVEautochessAI
- 专文：[EVEautochessAI-design / docs/AI_SELFPLAY.md](https://github.com/EveAutochess-develop/EVEautochessAI-design/blob/main/docs/AI_SELFPLAY.md)
- 本地目录：`H:\game_dev\EVEautochessAI-main`

## 本轮骨架

- `schema/`：commands、board_desc、genome、weights_table、telemetry、memory
- `orchestrator.py`：一代 20 席，留前 3
- `battle_stub.py`：`backend=cpu_stub`（可占位 `gpu_stub`），返回 schema 合法空包
- content：读游戏仓 `godot_project/data/`（见 `config.json`），不复制 Godot

```powershell
python orchestrator.py
```

写出 `samples/behavior.genome.json` 与 `samples/weights_table.json`。

完整战斗核 / CUDA / 金标对齐见 `golden_placeholder.md`，不在本轮。
