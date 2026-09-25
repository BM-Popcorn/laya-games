"""Playable Connect Four: human vs Laya, or Laya vs Laya.

    .venv\\Scripts\\python.exe connect4_demo.py --mode mock
    .venv\\Scripts\\python.exe connect4_demo.py --mode laya --players human-laya
    .venv\\Scripts\\python.exe connect4_demo.py --mode laya --players laya-laya
"""
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
import tkinter as tk
from tkinter import messagebox

import connect4_engine as engine
from connect4_laya import decide

ROOT = Path(__file__).parent
ROWS, COLS = engine.ROWS, engine.COLS
EMPTY, HUMAN, LAYA = 0, engine.HUMAN, engine.LAYA


class ConnectFour(tk.Tk):
    def __init__(self, mode, players):
        super().__init__()
        self.title("Laya Connect Four")
        self.mode, self.players = mode, players
        self.board = [[0] * COLS for _ in range(ROWS)]
        self.over = False
        self.moves = []
        self.turn = HUMAN
        self.info = tk.StringVar(value=self._prompt())
        tk.Label(self, textvariable=self.info, font=("Segoe UI", 13)).pack(pady=8)
        self.canvas = tk.Canvas(self, width=560, height=500, bg="#164e63", highlightthickness=0)
        self.canvas.pack(padx=12, pady=8)
        self.canvas.bind("<Button-1>", self.human_move)
        tk.Button(self, text="New game", command=self.new_game).pack(pady=(0, 12))
        self.draw()
        self.persist_state()
        if self.players == "laya-laya":
            self.after(250, self.laya_move)

    def _prompt(self):
        if self.players == "laya-laya":
            return "Laya (yellow) is thinking…"
        return "Your turn — click a column"

    def new_game(self):
        self.board = [[0] * COLS for _ in range(ROWS)]
        self.over, self.turn, self.moves = False, HUMAN, []
        self.info.set(self._prompt())
        self.draw()
        self.persist_state()
        if self.players == "laya-laya":
            self.after(250, self.laya_move)

    def draw(self):
        self.canvas.delete("all")
        for r in range(ROWS):
            for c in range(COLS):
                colour = {EMPTY: "#e2e8f0", HUMAN: "#f59e0b", LAYA: "#ef4444"}[self.board[r][c]]
                x, y = c * 80 + 10, r * 80 + 10
                self.canvas.create_oval(x, y, x + 60, y + 60, fill=colour, outline="#0f172a", width=2)

    def human_move(self, event):
        if self.over or self.players != "human-laya" or self.turn != HUMAN:
            return
        col = max(0, min(COLS - 1, event.x // 80))
        if board_full_column(self.board, col):
            return
        engine.drop(self.board, col, HUMAN)
        self.draw()
        self.moves.append({"player": "human", "column": col + 1, "at": datetime.now(UTC).isoformat()})
        self.persist_state()
        if self.finish(HUMAN, "You win!"):
            return
        if not engine.legal_columns(self.board):
            return self.finish(EMPTY, "Draw game")
        self.turn = LAYA
        self.info.set("Laya is thinking…")
        self.after(50, self.laya_move)

    def laya_move(self):
        try:
            proposal = decide(self.board, self.mode, self.turn)
            col = int(proposal["recommended_action"].split("_")[-1]) - 1
            if board_full_column(self.board, col):
                raise ValueError("Laya returned an illegal column")
            engine.drop(self.board, col, self.turn)
            self.draw()
            self.moves.append({
                "player": "laya",
                "colour": "yellow" if self.turn == HUMAN else "red",
                "column": col + 1,
                "decision": proposal,
                "at": datetime.now(UTC).isoformat(),
            })
            self.persist_state(proposal)
            confidence = proposal.get("confidence")
            text = f"Laya chose column {col + 1}"
            if isinstance(confidence, (int, float)):
                text += f" ({confidence:.0%} confidence)"
            self.info.set(text)
            if self.finish(self.turn, f"Laya ({'red' if self.turn == LAYA else 'yellow'}) wins!"):
                return
            if not engine.legal_columns(self.board):
                self.finish(EMPTY, "Draw game")
                return
            self.turn = HUMAN if self.turn == LAYA else LAYA
            if self.players == "laya-laya":
                side = "yellow" if self.turn == HUMAN else "red"
                self.info.set(f"Laya ({side}) is thinking…")
                self.after(250, self.laya_move)
            else:
                self.info.set("Your turn — click a column")
        except Exception as exc:
            self.info.set(f"Decision error: {exc}")

    def persist_state(self, latest=None):
        snapshot = {
            "game": "connect_four",
            "players": self.players,
            "turn": self.turn,
            "board": self.board,
            "moves": self.moves,
            "latest_decision": latest,
            "updated_at": datetime.now(UTC).isoformat(),
        }
        (ROOT / "connect4_state.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")

    def finish(self, player, text):
        found = engine.winner(self.board)
        if player == EMPTY or found == player:
            self.over = True
            self.info.set(text)
            messagebox.showinfo("Connect Four", text)
            return True
        return False


def board_full_column(board, col):
    return board[0][col] != EMPTY


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("mock", "laya"), default="mock", help="decision engine")
    parser.add_argument("--players", choices=("human-laya", "laya-laya"), default="human-laya")
    args = parser.parse_args()
    ConnectFour(args.mode, args.players).mainloop()
