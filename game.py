"""
game.py — Programmatic Minesweeper game engine.

Board cell values:
    UNKNOWN (-1) : not yet revealed
    MINE    (-2) : revealed mine (game over)
    0-8          : revealed cell with adjacent mine count
"""

import random
from typing import Iterator, Set, Tuple

UNKNOWN = -1
MINE = -2


class MinesweeperGame:
    """
    Self-contained Minesweeper game.

    Mines are placed on the first reveal, guaranteeing the opening cell and
    all its neighbours are safe (standard modern Minesweeper behaviour).
    """

    def __init__(self, rows: int, cols: int, total_mines: int) -> None:
        if total_mines >= rows * cols:
            raise ValueError("Too many mines for the board size.")
        self.rows = rows
        self.cols = cols
        self.total_mines = total_mines

        self.board: list[list[int]] = [[UNKNOWN] * cols for _ in range(rows)]
        self._mines: list[list[bool]] = [[False] * cols for _ in range(rows)]

        self.game_over: bool = False
        self.won: bool = False
        self._initialized: bool = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reveal(self, r: int, c: int) -> bool:
        """
        Reveal cell (r, c).

        Returns:
            True  — safe reveal (game continues or just won)
            False — mine hit (game over)
        """
        if self.game_over:
            return not self._mines[r][c]
        if self.board[r][c] != UNKNOWN:
            return True

        if not self._initialized:
            self._place_mines(safe_r=r, safe_c=c)

        if self._mines[r][c]:
            self.board[r][c] = MINE
            self.game_over = True
            return False

        self._flood_reveal(r, c)
        self._check_win()
        return True

    def flag(self, r: int, c: int) -> None:
        """Mark a cell as a known mine (does not affect solver logic here)."""
        if self.board[r][c] == UNKNOWN:
            self.board[r][c] = MINE

    def print_board(self, reveal_all: bool = False) -> None:
        """Pretty-print the current board state."""
        symbols = {UNKNOWN: ".", MINE: "*", 0: " "}
        for r in range(self.rows):
            row = ""
            for c in range(self.cols):
                val = self.board[r][c]
                if val == UNKNOWN and reveal_all and self._mines[r][c]:
                    row += "X"
                else:
                    row += symbols.get(val, str(val))
            print(row)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _neighbors(self, r: int, c: int) -> Iterator[Tuple[int, int]]:
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    yield nr, nc

    def _place_mines(self, safe_r: int, safe_c: int) -> None:
        safe: Set[Tuple[int, int]] = {(safe_r, safe_c)}
        safe.update(self._neighbors(safe_r, safe_c))

        candidates = [
            (r, c)
            for r in range(self.rows)
            for c in range(self.cols)
            if (r, c) not in safe
        ]
        for r, c in random.sample(candidates, self.total_mines):
            self._mines[r][c] = True
        self._initialized = True

    def _adj_mine_count(self, r: int, c: int) -> int:
        return sum(self._mines[nr][nc] for nr, nc in self._neighbors(r, c))

    def _flood_reveal(self, start_r: int, start_c: int) -> None:
        """BFS flood-fill reveal starting from a safe, unrevealed cell."""
        queue: list[Tuple[int, int]] = [(start_r, start_c)]
        seen: Set[Tuple[int, int]] = set()
        while queue:
            r, c = queue.pop()
            if (r, c) in seen:
                continue
            seen.add((r, c))
            count = self._adj_mine_count(r, c)
            self.board[r][c] = count
            if count == 0:
                for nr, nc in self._neighbors(r, c):
                    if self.board[nr][nc] == UNKNOWN and (nr, nc) not in seen:
                        queue.append((nr, nc))

    def _check_win(self) -> None:
        unknown = sum(
            self.board[r][c] == UNKNOWN
            for r in range(self.rows)
            for c in range(self.cols)
        )
        confirmed_mines = sum(
            self.board[r][c] == MINE
            for r in range(self.rows)
            for c in range(self.cols)
        )
        remaining = self.total_mines - confirmed_mines
        if unknown == remaining:
            self.won = True
            self.game_over = True
