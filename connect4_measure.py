"""Measure how often Laya's raw column matches the engine on held-out positions.

The comparison uses the model's choice, before the search override.
Reports overall agreement plus a forced (one-hot) vs soft (multi-way) split.
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


def is_forced(target):
    """True when the engine label is a one-hot forced win/loss."""
    vals = list(target.values())
    return max(vals) >= 0.999 and sum(1 for v in vals if v > 0) == 1


def ply(board):
    return sum(1 for row in board for cell in row if cell)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("holdout",), default="holdout")
    parser.add_argument("--checkpoint", default="", help="folder with model.safetensors; default uses load_agent")
    parser.add_argument("--limit", type=int, default=0, help="max rows; 0 uses the full holdout")
    parser.add_argument("--miss-examples", type=int, default=5, help="print this many missed forced positions")
    args = parser.parse_args()
    path = HOLDOUT
    rows = load_rows(path, args.limit)
    if not rows:
        raise SystemExit(f"No rows in {path}")
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
    forced_hits = forced_n = 0
    soft_hits = soft_n = 0
    missed_forced = []

    for index, row in enumerate(rows, start=1):
        state, actions = connect4_laya.state_and_actions(row["board"], row["player"])
        answer = agent.predict(state, connect4_laya.questions_for(actions))["answers"]["next_move"]
        choice = answer["choice"]
        hit = agreed(choice, row["target"])
        if hit:
            hits += 1
        if is_forced(row["target"]):
            forced_n += 1
            if hit:
                forced_hits += 1
            elif len(missed_forced) < args.miss_examples:
                missed_forced.append({
                    "ply": ply(row["board"]),
                    "player": row["player"],
                    "label": row["label"],
                    "choice": choice,
                })
        else:
            soft_n += 1
            if hit:
                soft_hits += 1
        if index % 50 == 0 or index == len(rows):
            print(f"{index}/{len(rows)} agreement {hits / index:.3f}")

    n = len(rows)
    print(f"checkpoint: {folder}")
    print(f"top-1 agreement with the engine: {hits}/{n} = {hits / max(1, n):.3f}")
    print(
        f"forced (one-hot): {forced_hits}/{forced_n} = "
        f"{forced_hits / max(1, forced_n):.3f}"
    )
    print(
        f"soft (multi-way): {soft_hits}/{soft_n} = "
        f"{soft_hits / max(1, soft_n):.3f}"
    )
    if missed_forced:
        print("missed forced examples:")
        for miss in missed_forced:
            print(
                f"  ply {miss['ply']} player {miss['player']} "
                f"engine {miss['label']} model {miss['choice']}"
            )


if __name__ == "__main__":
    main()
