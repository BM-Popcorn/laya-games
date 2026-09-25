# Laya integration

This project uses the open-source [Laya](https://github.com/NandhaKishorM/laya) typed-decision checkpoint. Laya chooses one legal Connect Four column. The local search engine decides whether that column is good enough to play.

## What this project adds

- `connect4_engine.py` scores every legal column with alpha-beta negamax.
- `connect4_laya.py` asks Laya for a column and applies the engine as a safety layer. Search scores are not written into the prompt.
- `connect4_demo.py` is the desktop board.
- `decisions/connect4_latest.json` stores the latest proposal and the column that was played.
- `decisions/connect4_audit.jsonl` stores the move history.
- `connect4_dashboard.html` displays the board, proposal, engine decision, and confidence.

## Model

On the first `--mode laya` move, the loader uses `models/connect4` when that fine-tune exists, otherwise `models/typed-decisions`. The typed-decisions checkpoint is a general decision model. Fine-tuning it on engine-labelled positions is what teaches it Connect Four.

## Modes

- `human-laya`: a person plays yellow and Laya plays red.
- `laya-laya`: Laya plays both colours, each move still checked by the engine.
- `mock`: the engine plays, with no model inference.
