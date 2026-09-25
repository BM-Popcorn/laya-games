"""Laya proposals for Connect Four, checked by the search engine.

The model sees the board and the legal columns only. Search scores stay out of
the prompt so a fine-tune cannot learn to copy them. The engine still replaces
a proposal that loses by force or falls well short of the best move.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

import connect4_engine as engine

ROOT = Path(__file__).parent
DECISIONS = ROOT / "decisions"
MODELS = ROOT / "models"
INSTRUCTION = (
    "Choose exactly one legal Connect Four column. "
    "Prefer a winning move, otherwise block the opponent, otherwise create a strong threat."
)


def checkpoint_dir():
    """Prefer a Connect Four fine-tune, otherwise the typed-decisions checkpoint."""
    tuned = MODELS / "connect4"
    if (tuned / "model.safetensors").is_file() and (tuned / "rl_agent_config.json").is_file():
        return tuned, None
    base = MODELS / "typed-decisions"
    if (base / "model.safetensors").is_file():
        return MODELS, "typed-decisions"
    return None, None


@lru_cache(maxsize=1)
def load_agent():
    import laya
    folder, subfolder = checkpoint_dir()
    if folder is None:
        from huggingface_hub import snapshot_download
        folder = snapshot_download(
            "convaiinnovations/laya",
            repo_type="model",
            allow_patterns=["typed-decisions/*"],
            local_dir=MODELS,
        )
        subfolder = "typed-decisions"
    return laya.load(str(folder), subfolder=subfolder)


def state_and_actions(board, player):
    legal = sorted(c for c in range(engine.COLS) if not board[0][c])
    turn = "laya_red" if player == engine.LAYA else "laya_yellow"
    actions = [
        {"id": f"column_{c + 1}", "description": f"Drop a counter in column {c + 1}"}
        for c in legal
    ]
    state = {
        "game": {
            "name": "connect_four",
            "rows": engine.ROWS,
            "columns": engine.COLS,
            "turn": turn,
            "board": board,
        },
        "available_actions": actions,
        "goal": {
            "summary": "Win Connect Four by making four counters in a row; block an immediate opponent win."
        },
    }
    return state, actions


def questions_for(actions):
    return {
        "next_move": {
            "type": "choice",
            "instructions": INSTRUCTION,
            "criteria": {a["id"]: a["description"] for a in actions},
        }
    }


def _column_from_choice(choice):
    return int(str(choice).split("_")[-1]) - 1


def decide(board, mode="laya", player=engine.LAYA, depth=engine.DEFAULT_DEPTH):
    """Return the logged decision. `action` is the column that should be played."""
    state, actions = state_and_actions(board, player)
    if not actions:
        raise ValueError("no legal moves")
    scores = engine.score_columns(board, player, depth)
    answer = None
    if mode == "mock":
        model_col = None
        confidence = 1.0
    else:
        answer = load_agent().predict(state, questions_for(actions))["answers"]["next_move"]
        model_col = _column_from_choice(answer["choice"])
        confidence = answer.get("confidence")
    final = engine.choose(board, model_col=model_col, player=player, depth=depth, scores=scores)
    model_action = f"column_{model_col + 1}" if model_col is not None else None
    if mode == "mock":
        model_action = f"column_{final['column'] + 1}"
        reason = "mock policy follows the search engine"
        overridden = False
        column = final["column"]
    else:
        reason = final["reason"]
        overridden = final["overridden"]
        column = final["column"]
    result = {
        "game": "connect4",
        "generated_at": datetime.now(UTC).isoformat(),
        "mode": mode,
        "player": player,
        "action": f"column_{column + 1}",
        "recommended_action": f"column_{column + 1}",
        "model_action": model_action,
        "overridden": overridden,
        "reason": reason,
        "confidence": confidence,
        "state": state,
        "legal_actions": actions,
        "legal_columns": [c + 1 for c in sorted(scores)],
        "board": board,
        "strategy": {
            "depth": depth,
            "scores": {f"column_{c + 1}": v for c, v in sorted(scores.items())},
            "best_column": final["best_column"] + 1,
        },
        "raw_answer": answer,
    }
    DECISIONS.mkdir(exist_ok=True)
    (DECISIONS / "connect4_latest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (DECISIONS / "connect4_audit.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result) + "\n")
    return result
