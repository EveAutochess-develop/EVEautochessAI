# EVEautochessAI · 无头互训农场

独立进程：世代编排、CUDA 批对局推理、导出 `behavior.genome` + T0–Tn。  
**不进** Godot 工程。运行时**不调** LLM API。

- GitHub：https://github.com/EveAutochess-develop/EVEautochessAI（本里程碑不推）
- 专文：[EVEautochessAI-design / docs/AI_SELFPLAY.md](https://github.com/EveAutochess-develop/EVEautochessAI-design/blob/main/docs/AI_SELFPLAY.md)
- 本地目录：`H:\game_dev\EVEautochessAI-main`

## 运行

Cursor 集成终端：进程入口把 stdout 设成 UTF-8（避免 GBK 乱码）。PowerShell 先 `chcp 65001`。停训写 `samples/farm.stop` 或终端 Ctrl+C（当代结束再退）。不要对 pid `SIGTERM`。

```powershell
chcp 65001 | Out-Null
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
.\.venv\Scripts\python -X utf8 -u -m eveac_ai.match20
```

农场目录 `.venv` 装 **CUDA PyTorch**（`cu128`）。`config.json`：`battle_backend` 为 `gpu`（无卡回退 CPU 核）。开局 **两轮选泰坦**（盲选 → 公开各族人数 → 可维持或改选），人格不烙泰坦。

日志分家（各 **500MB** 上限）：`samples/logs/diag.log` 仅 debug；`samples/logs/replay.txt` 开场快照 + 当场结果（编号摆放表含等候席、星级装备、经济；PVE 写 L0 联盟布局）。中盘机器可读：`samples/match_checkpoint.json` + `samples/logs/match_checkpoints/`。

## 布局

- `eveac_ai/`：content、formulas、ship、cpu/gpu kernel、titan_draft、bootstrap、orchestrator
- `priors/llm_bootstrap.genome.json`：五泰坦切片 + `titan_pick` + 五模式
- `schema/`：契约 JSON Schema
- `tests/`：公式对照 + 冻结 1v1 + 两轮选泰坦
- `battle_stub.py`：空核（`cpu_stub` / `gpu_stub`）

Content 读游戏仓 `godot_project/data/`（见 `config.json`），不复制 Godot。

GPU 核：批桌 Turret/打层/锁定/电容；导弹即时 DR（无弹体）。金标见 `golden_placeholder.md`。
