# Laya Connect Four

Connect Four driven by [Laya](https://github.com/NandhaKishorM/laya), a typed-decision model. Laya proposes a legal column. `connect4_engine.py` scores every column with depth-7 alpha-beta search and keeps the proposal when it is close to the best move.

## Play

```powershell
.\.venv\Scripts\python.exe connect4_demo.py --mode laya --players human-laya
.\.venv\Scripts\python.exe connect4_demo.py --mode laya --players laya-laya
```

`--mode mock` uses the search engine and does not load the model.

## Dashboard

```powershell
.\.venv\Scripts\python.exe dashboard_server.py
```

Open `http://127.0.0.1:8765/play_dashboard.html` to play, or `http://127.0.0.1:8765/connect4_dashboard.html` for the latest board, proposal, and search scores.

Each move is appended to `decisions/connect4_audit.jsonl`. The latest decision is `decisions/connect4_latest.json`. The desktop game writes `connect4_state.json`.

## Train on the engine

Generate engine-labelled positions, fine-tune the typed-decision checkpoint, then measure how often Laya's raw column matches the engine on positions it did not train on:

```powershell
.\.venv\Scripts\python.exe -m pip install bitsandbytes
.\.venv\Scripts\python.exe connect4_train.py
```

The labelled positions are already in `data/connect4_train.jsonl` (3,577) and `data/connect4_holdout.jsonl` (423). Training starts from `models/typed-decisions`. On a CUDA GPU the encoder is updated. The micro-batch defaults to 1 with 8 accumulation steps so an 8 GB card can hold the model when 8-bit AdamW is installed.

Every 50 optimizer steps the script prints held-out agreement on the same 40 positions used for the earlier score (baseline 20/40), appends that line to `data/connect4_progress.jsonl`, and writes `models/connect4`. Stop the run when that number stops climbing. The measurement is the model's own column, before the engine override.

More detail is in [docs/LAYA.md](docs/LAYA.md).
