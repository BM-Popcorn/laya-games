"""Generate Connect Four positions labelled by the search engine.

Self-play mixes the engine's move with a random legal column so the set is not
only perfect games. Each unique position is stored once, then forced (one-hot)
train rows are upsampled so tactical positions are denser in the train set. A
fixed fraction of unique positions is held out and never used for training.
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


def is_forced_row(row):
    vals = list(row["target"].values())
    return max(vals) >= 0.999 and sum(1 for v in vals if v > 0) == 1


def is_immediate_tactic(board, player):
    """Immediate win for player, or a must-block against the opponent."""
    own = [c for c in engine.legal_columns(board) if engine.wins_now(board, c, player)]
    if own:
        return True
    opp = engine.HUMAN if player == engine.LAYA else engine.LAYA
    return any(engine.wins_now(board, c, opp) for c in engine.legal_columns(board))


def play_game_immediate(args):
    """Self-play that only returns immediate win/block positions (one-hot tactics)."""
    seed, depth, random_move_rate = args
    rng = random.Random(seed)
    board = [[0] * engine.COLS for _ in range(engine.ROWS)]
    player = engine.HUMAN
    rows = []
    seen = set()
    while engine.legal_columns(board) and not engine.winner(board):
        if is_immediate_tactic(board, player):
            key = (engine._pack(board), player)
            if key not in seen:
                seen.add(key)
                scores = engine.score_columns(board, player, depth)
                rows.append(_row(board, player, scores))
        scores = engine.score_columns(board, player, depth)
        legal = list(scores)
        col = rng.choice(legal) if rng.random() < random_move_rate else engine._best(scores)
        engine.drop(board, col, player)
        player = engine.LAYA if player == engine.HUMAN else engine.HUMAN
    return rows


def append_immediate_tactics(count, depth, games, workers, random_move_rate):
    """Append unique immediate win/block positions to the training set only."""
    existing_rows = []
    seen_keys = set()
    if TRAIN_PATH.is_file():
        with TRAIN_PATH.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    existing_rows.append(row)
                    seen_keys.add((engine._pack(row["board"]), row["player"]))
    kept = {}
    seed = 100_000
    with Pool(workers) as pool:
        while len(kept) < count and seed < 100_000 + games:
            wave = [(seed + i, depth, random_move_rate) for i in range(workers)]
            seed += workers
            for rows in pool.map(play_game_immediate, wave):
                for row in rows:
                    key = (engine._pack(row["board"]), row["player"])
                    if key not in seen_keys and key not in kept:
                        kept[key] = row
            if len(kept) % 200 < workers or len(kept) >= count:
                print(f"  immediate tactics {len(kept)}/{count} after {seed - 100_000} games", flush=True)
    added = list(kept.values())[:count]
    # Upsample: each immediate tactic x4 into train; keep existing rows as-is
    train_rows = existing_rows + added * 4
    _write(TRAIN_PATH, train_rows)
    print(f"Appended {len(added)} unique immediate tactics (x4) -> {len(train_rows)} train rows")


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


def _upsample_forced(train, copies):
    """Replicate forced train rows so tactics are denser without touching holdout."""
    if copies <= 1:
        return train
    forced = [row for row in train if is_forced_row(row)]
    extra = []
    for _ in range(copies - 1):
        extra.extend(forced)
    return train + extra


def generate(positions, depth, games, workers, random_move_rate, holdout_rate, forced_copies):
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
            if seed % max(workers * 10, 1) == 0 or len(kept) >= positions:
                print(f"  unique positions {len(kept)}/{positions} after {seed} games")
    rows = list(kept.values())[:positions]
    train, holdout = [], []
    for row in rows:
        (holdout if _holdout(row, holdout_rate) else train).append(row)
    forced_unique = sum(1 for row in train if is_forced_row(row))
    train = _upsample_forced(train, forced_copies)
    _write(TRAIN_PATH, train)
    _write(HOLDOUT_PATH, holdout)
    print(f"Wrote {len(train)} training positions to {TRAIN_PATH} "
          f"({forced_unique} unique forced x{forced_copies})")
    print(f"Wrote {len(holdout)} held-out positions to {HOLDOUT_PATH}")


def _write(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--positions", type=int, default=28000,
                        help="unique positions before holdout split (train+holdout)")
    parser.add_argument("--games", type=int, default=6000)
    parser.add_argument("--depth", type=int, default=engine.DEFAULT_DEPTH)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--random-move-rate", type=float, default=0.28)
    parser.add_argument("--holdout-rate", type=float, default=0.1)
    parser.add_argument("--forced-copies", type=int, default=3,
                        help="replicate each forced train row this many times (1 = no upsample)")
    parser.add_argument("--append-immediate", type=int, default=0,
                        help="if >0, append this many unique immediate win/block positions to train only")
    args = parser.parse_args()
    if args.append_immediate:
        append_immediate_tactics(
            args.append_immediate,
            args.depth,
            args.games,
            args.workers,
            args.random_move_rate,
        )
    else:
        generate(
            args.positions,
            args.depth,
            args.games,
            args.workers,
            args.random_move_rate,
            args.holdout_rate,
            args.forced_copies,
        )
