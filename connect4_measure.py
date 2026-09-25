"""Measure how often Laya's raw column matches the engine on held-out positions.

The comparison uses the model's choice, before the search override.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import connect4_laya

ROOT = Path(__file__).parent
HOLDOUT = ROOT / "data" / "connect4_holdout.jsonl"


def load_rows(path, limit):
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def agreed(choice, target):
    best = max(target.values())
    return target.get(choice, 0.0) == best and best > 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("holdout",), default="holdout")
    parser.add_argument("--checkpoint", default="", help="unused; load_agent picks models/connect4 when it exists")
    parser.add_argument("--limit", type=int, default=80)
    args = parser.parse_args()
    path = HOLDOUT
    rows = load_rows(path, args.limit)
    if args.checkpoint:
        import laya
        folder = Path(args.checkpoint)
        if not (folder / "model.safetensors").is_file():
            raise SystemExit(f"No checkpoint at {folder}")
        agent = laya.load(str(folder.resolve()))
    else:
        agent = connect4_laya.load_agent()
        folder, _ = connect4_laya.checkpoint_dir()
    hits = 0
    for index, row in enumerate(rows, start=1):
        state, actions = connect4_laya.state_and_actions(row["board"], row["player"])
        answer = agent.predict(state, connect4_laya.questions_for(actions))["answers"]["next_move"]
        if agreed(answer["choice"], row["target"]):
            hits += 1
        if index % 10 == 0 or index == len(rows):
            print(f"{index}/{len(rows)} agreement {hits / index:.3f}")
    print(f"checkpoint: {folder}")
    print(f"top-1 agreement with the engine: {hits}/{len(rows)} = {hits / max(1, len(rows)):.3f}")


if __name__ == "__main__":
    main()
