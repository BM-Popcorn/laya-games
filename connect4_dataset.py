"""Generate Connect Four positions labelled by the search engine.

Self-play mixes the engine's move with a random legal column so the set is not
only perfect games. Each position is stored once. A fixed fraction is held out
and never used for training.
"""
from __future__ import annotations

import argparse
import json
import random
from multiprocessing import Pool
from pathlib import Path

import connect4_engine as engine

ROOT = Path(__file__).parent
DATA = ROOT / "data"
TRAIN_PATH = DATA / "connect4_train.jsonl"
HOLDOUT_PATH = DATA / "connect4_holdout.jsonl"


def _row(board, player, scores):
    target = engine.target_distribution(scores)
    legal = sorted(target)
    return {
        "board": [row[:] for row in board],
        "player": player,
        "label": f"column_{engine._best(scores) + 1}",
        "target": {f"column_{c + 1}": target[c] for c in legal},
        "scores": {f"column_{c + 1}": scores[c] for c in legal},
    }


def play_game(args):
    seed, depth, random_move_rate = args
    rng = random.Random(seed)
    board = [[0] * engine.COLS for _ in range(engine.ROWS)]
    player = engine.HUMAN
    rows = []
    seen = set()
    while engine.legal_columns(board) and not engine.winner(board):
        scores = engine.score_columns(board, player, depth)
        key = (engine._pack(board), player)
        if key not in seen:
            seen.add(key)
            rows.append(_row(board, player, scores))
        legal = list(scores)
        col = rng.choice(legal) if rng.random() < random_move_rate else engine._best(scores)
        engine.drop(board, col, player)
        player = engine.LAYA if player == engine.HUMAN else engine.HUMAN
    return rows


def _holdout(row, rate):
    key = (engine._pack(row["board"]), row["player"], 17)
    return (key[0] ^ (key[1] * 0x9E3779B1)) % 10_000 < int(rate * 10_000)


def generate(positions, depth, games, workers, random_move_rate, holdout_rate):
    DATA.mkdir(exist_ok=True)
    kept = {}
    seed = 0
    with Pool(workers) as pool:
        while len(kept) < positions and seed < games:
            wave = [(seed + i, depth, random_move_rate) for i in range(workers)]
            seed += workers
            for rows in pool.map(play_game, wave):
                for row in rows:
                    kept[(engine._pack(row["board"]), row["player"])] = row
    rows = list(kept.values())[:positions]
    train, holdout = [], []
    for row in rows:
        (holdout if _holdout(row, holdout_rate) else train).append(row)
    _write(TRAIN_PATH, train)
    _write(HOLDOUT_PATH, holdout)
    print(f"Wrote {len(train)} training positions to {TRAIN_PATH}")
    print(f"Wrote {len(holdout)} held-out positions to {HOLDOUT_PATH}")


def _write(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--positions", type=int, default=4000)
    parser.add_argument("--games", type=int, default=800)
    parser.add_argument("--depth", type=int, default=engine.DEFAULT_DEPTH)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--random-move-rate", type=float, default=0.2)
    parser.add_argument("--holdout-rate", type=float, default=0.1)
    args = parser.parse_args()
    generate(args.positions, args.depth, args.games, args.workers, args.random_move_rate, args.holdout_rate)
