"""Connect Four search engine used as the strategy layer behind Laya.

Laya proposes a column; this module scores every legal column with an
alpha-beta negamax search and decides whether the proposal is acceptable.
Board convention matches the dashboard: board[0] is the top row, 0 empty,
1 human, 2 Laya.
"""
from __future__ import annotations

ROWS, COLS = 6, 7
HUMAN, LAYA = 1, 2
WIN = 100_000
FORCED = 50_000          # |score| above this means a forced win/loss was found
DEFAULT_DEPTH = 7
DEFAULT_TOLERANCE = 12   # how far below the best move Laya's pick may score and still be accepted
ORDER = sorted(range(COLS), key=lambda c: abs(c - COLS // 2))
_EXACT, _LOWER, _UPPER = 0, 1, 2
_TT = {}
_TT_CAP = 250_000

_WINDOWS = []
for _r in range(ROWS):
    for _c in range(COLS):
        for _dr, _dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            cells = [(_r + _dr * i, _c + _dc * i) for i in range(4)]
            if all(0 <= r < ROWS and 0 <= c < COLS for r, c in cells):
                _WINDOWS.append(cells)


def legal_columns(board):
    return [c for c in ORDER if not board[0][c]]


def drop(board, col, player):
    """Place `player` in `col`. Mutates `board` and returns the landing row."""
    r = _land_row(board, col)
    if r < 0:
        raise ValueError("column is full")
    board[r][col] = player
    return r


def winner(board):
    """Return the player who has four in a row, or 0."""
    for cells in _WINDOWS:
        vals = [board[r][c] for r, c in cells]
        if vals[0] and vals[0] == vals[1] == vals[2] == vals[3]:
            return vals[0]
    return 0


def _pack(board):
    key = 0
    for row in board:
        for v in row:
            key = key * 3 + v
    return key


def _land_row(board, col):
    for r in range(ROWS - 1, -1, -1):
        if not board[r][col]:
            return r
    return -1


def _makes_four(board, r, c, p):
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        n = 1
        for sign in (1, -1):
            rr, cc = r + dr * sign, c + dc * sign
            while 0 <= rr < ROWS and 0 <= cc < COLS and board[rr][cc] == p:
                n += 1
                rr += dr * sign
                cc += dc * sign
        if n >= 4:
            return True
    return False


def wins_now(board, col, p):
    """True if p playing col wins immediately."""
    r = _land_row(board, col)
    if r < 0:
        return False
    board[r][col] = p
    won = _makes_four(board, r, col, p)
    board[r][col] = 0
    return won


def _evaluate(board, p):
    """Static score from p's point of view: open windows, centre control."""
    q = HUMAN if p == LAYA else LAYA
    score = sum(3 for r in range(ROWS) if board[r][COLS // 2] == p) - sum(3 for r in range(ROWS) if board[r][COLS // 2] == q)
    for cells in _WINDOWS:
        mine = theirs = 0
        for r, c in cells:
            v = board[r][c]
            if v == p:
                mine += 1
            elif v == q:
                theirs += 1
        if mine and theirs:
            continue
        if mine == 3:
            score += 5
        elif mine == 2:
            score += 2
        if theirs == 3:
            score -= 6
        elif theirs == 2:
            score -= 2
    return score


def _negamax(board, depth, alpha, beta, p, filled):
    if filled == ROWS * COLS:
        return 0
    orig_alpha = alpha
    key = (_pack(board), p, depth)
    hit = _TT.get(key)
    if hit is not None:
        flag, score = hit
        if flag == _EXACT:
            return score
        if flag == _LOWER and score > alpha:
            alpha = score
        elif flag == _UPPER and score < beta:
            beta = score
        if alpha >= beta:
            return score
    bound = orig_alpha
    q = HUMAN if p == LAYA else LAYA
    moves = legal_columns(board)
    for c in moves:                       # immediate win for the side to move
        if wins_now(board, c, p):
            score = WIN + depth
            _remember(key, _EXACT, score)
            return score
    if depth == 0:
        score = _evaluate(board, p)
        _remember(key, _EXACT, score)
        return score
    best = -WIN - 1
    for c in moves:
        r = _land_row(board, c)
        board[r][c] = p
        score = -_negamax(board, depth - 1, -beta, -alpha, q, filled + 1)
        board[r][c] = 0
        if score > best:
            best = score
        if best > alpha:
            alpha = best
        if alpha >= beta:
            break
    if best <= bound:
        flag = _UPPER
    elif best >= beta:
        flag = _LOWER
    else:
        flag = _EXACT
    _remember(key, flag, best)
    return best


def _remember(key, flag, score):
    if len(_TT) >= _TT_CAP:
        _TT.clear()
    _TT[key] = (flag, score)


def score_columns(board, player=LAYA, depth=DEFAULT_DEPTH):
    """Return {column(0-based): score} from `player`'s point of view."""
    board = [row[:] for row in board]
    filled = sum(1 for row in board for v in row if v)
    opp = HUMAN if player == LAYA else LAYA
    scores = {}
    for c in legal_columns(board):
        r = _land_row(board, c)
        board[r][c] = player
        if _makes_four(board, r, c, player):
            scores[c] = WIN + depth
        else:
            scores[c] = -_negamax(board, depth - 1, -WIN - 1, WIN + 1, opp, filled + 1)
        board[r][c] = 0
    return scores


def _best(scores):
    top = max(scores.values())
    return next(c for c in ORDER if c in scores and scores[c] == top)   # centre-first tiebreak


def target_distribution(scores, tolerance=DEFAULT_TOLERANCE):
    """Soft label over legal columns, matching the move `choose` would play.

    Forced wins and losses are one-hot on the engine's tie-break. Quiet
    positions share probability across every column within `tolerance`.
    """
    best_col = _best(scores)
    best = scores[best_col]
    if abs(best) >= FORCED:
        return {c: 1.0 if c == best_col else 0.0 for c in scores}
    near = [c for c in scores if best - scores[c] <= tolerance]
    share = 1.0 / len(near)
    return {c: (share if c in near else 0.0) for c in scores}


def choose(board, model_col=None, player=LAYA, depth=DEFAULT_DEPTH, tolerance=DEFAULT_TOLERANCE, scores=None):
    """Decide the final column, given Laya's proposal (0-based, or None).

    Returns a dict with the final 0-based column, a human-readable reason,
    whether the proposal was overridden, and the per-column scores.
    """
    scores = scores or score_columns(board, player, depth)
    if not scores:
        raise ValueError("no legal moves")
    opp = HUMAN if player == LAYA else LAYA
    best_col = _best(scores)
    best = scores[best_col]
    threats = [c for c in legal_columns(board) if wins_now(board, c, opp)]
    own_wins = [c for c in legal_columns(board) if wins_now(board, c, player)]

    def label(col):
        s = scores[col]
        if col in own_wins:
            return "take the winning move"
        if col in threats and s > -FORCED:
            return "block the opponent's winning move"
        if s >= FORCED:
            return "forced win found by search"
        if s <= -FORCED:
            return "every move loses by force; this delays it longest"
        return f"search prefers this column (score {s} vs {scores.get(model_col, 'n/a')} for Laya's pick)"

    if model_col is not None and model_col in scores:
        gap = best - scores[model_col]
        limit = 0 if abs(best) >= FORCED else tolerance
        if gap <= limit:
            return {"column": model_col, "overridden": False, "scores": scores, "best_column": best_col,
                    "reason": "accepted Laya proposal (search agrees)"}
        why = label(best_col)
        if scores[model_col] <= -FORCED and best > -FORCED:
            why = f"Laya's move loses by force; {why}"
        return {"column": best_col, "overridden": True, "scores": scores, "best_column": best_col,
                "reason": f"override: {why}"}
    return {"column": best_col, "overridden": False, "scores": scores, "best_column": best_col,
            "reason": f"engine choice: {label(best_col)}"}
