"""
solver.py — Monte Carlo Minesweeper solver.

Algorithm
---------
For each of N samples:
  1. Extract constraints from revealed numbered cells.
  2. Partition unknown cells into:
       boundary — adjacent to ≥1 revealed number (constrained)
       interior — no revealed neighbours  (only global mine count applies)
  3. Generate a consistent boundary assignment via randomised backtracking.
  4. Distribute the remaining mines uniformly at random over interior cells.
  5. Tally the mine hit for every cell in this sample.

After N valid samples, the mine probability of cell x is:
    P(mine | x) ≈ hits[x] / valid_samples

The safest move is argmin over unknown cells.
"""

import random
from collections import defaultdict
from typing import Dict, Iterator, List, Optional, Set, Tuple

from game import MINE, UNKNOWN

# Type aliases
Cell = Tuple[int, int]
Constraint = Tuple[List[Cell], int]   # (unknown_neighbours, mines_still_needed)
Assignment = Dict[Cell, int]           # cell → 0 or 1


class MonteCarloSolver:
    """
    Monte Carlo mine-probability estimator for Minesweeper.

    Parameters
    ----------
    n_samples : int
        Number of random consistent completions to generate.
        Higher values improve accuracy at the cost of runtime.
    max_bt_attempts : int
        Maximum backtracking attempts per sample before giving up and
        trying a fresh shuffle. Prevents long stalls on tightly constrained
        boards.
    """

    def __init__(self, n_samples: int = 500, max_bt_attempts: int = 5000) -> None:
        self.n_samples = n_samples
        self.max_bt_attempts = max_bt_attempts

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_probabilities(
        self,
        board: list[list[int]],
        rows: int,
        cols: int,
        total_mines: int,
    ) -> Dict[Cell, float]:
        """
        Estimate mine probability for every unknown cell.

        Returns
        -------
        Dict mapping (r, c) → estimated mine probability in [0, 1].
        Returns an empty dict if no unknown cells remain.
        """
        unknown_cells = self._unknown_cells(board, rows, cols)
        if not unknown_cells:
            return {}

        confirmed_mines = sum(
            board[r][c] == MINE for r in range(rows) for c in range(cols)
        )
        remaining = total_mines - confirmed_mines

        constraints = self._extract_constraints(board, rows, cols)
        boundary, interior = self._partition(unknown_cells, constraints)

        # Fast path: no constraints at all (pure interior board)
        if not constraints:
            p = remaining / len(unknown_cells)
            return {cell: p for cell in unknown_cells}

        # Bounds on how many boundary cells can be mines
        min_b = max(0, remaining - len(interior))
        max_b = min(remaining, len(boundary))

        mine_hits: Dict[Cell, int] = defaultdict(int)
        valid = 0

        for _ in range(self.n_samples):
            b_assign = self._sample_boundary(boundary, constraints, min_b, max_b)
            if b_assign is None:
                continue

            b_mines = sum(b_assign.values())
            i_mines = remaining - b_mines
            if i_mines < 0 or i_mines > len(interior):
                continue

            i_mine_set: Set[Cell] = (
                set(random.sample(interior, i_mines)) if i_mines > 0 else set()
            )

            valid += 1
            for cell, val in b_assign.items():
                mine_hits[cell] += val
            for cell in interior:
                mine_hits[cell] += int(cell in i_mine_set)

        if valid == 0:
            # Fell back: uniform prior
            p = remaining / len(unknown_cells)
            return {cell: p for cell in unknown_cells}

        return {cell: mine_hits[cell] / valid for cell in unknown_cells}

    def best_move(
        self,
        board: list[list[int]],
        rows: int,
        cols: int,
        total_mines: int,
    ) -> Optional[Cell]:
        """
        Return the cell with the lowest estimated mine probability.

        Returns None if no unknown cells exist.
        """
        probs = self.get_probabilities(board, rows, cols, total_mines)
        if not probs:
            return None

        best = min(probs, key=probs.get)   # type: ignore[arg-type]
        return best

    # ------------------------------------------------------------------
    # Constraint extraction
    # ------------------------------------------------------------------

    def _unknown_cells(self, board: list[list[int]], rows: int, cols: int) -> List[Cell]:
        return [
            (r, c)
            for r in range(rows)
            for c in range(cols)
            if board[r][c] == UNKNOWN
        ]

    def _neighbors(self, r: int, c: int, rows: int, cols: int) -> Iterator[Cell]:
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    yield nr, nc

    def _extract_constraints(
        self, board: list[list[int]], rows: int, cols: int
    ) -> List[Constraint]:
        """
        For every revealed numbered cell, build a constraint:
            (list of unknown neighbours, mines still needed among them)
        Only add constraints where at least one unknown neighbour exists.
        """
        constraints: List[Constraint] = []
        for r in range(rows):
            for c in range(cols):
                val = board[r][c]
                if val < 0:
                    continue  # UNKNOWN or MINE — not a clue cell
                unknown_nbrs: List[Cell] = []
                mine_nbrs = 0
                for nr, nc in self._neighbors(r, c, rows, cols):
                    if board[nr][nc] == UNKNOWN:
                        unknown_nbrs.append((nr, nc))
                    elif board[nr][nc] == MINE:
                        mine_nbrs += 1
                needed = val - mine_nbrs
                if unknown_nbrs:
                    constraints.append((unknown_nbrs, needed))
        return constraints

    def _partition(
        self, unknown_cells: List[Cell], constraints: List[Constraint]
    ) -> Tuple[List[Cell], List[Cell]]:
        """Split unknown cells into boundary and interior."""
        boundary_set: Set[Cell] = set()
        for nbrs, _ in constraints:
            boundary_set.update(nbrs)
        boundary = [c for c in unknown_cells if c in boundary_set]
        interior = [c for c in unknown_cells if c not in boundary_set]
        return boundary, interior

    # ------------------------------------------------------------------
    # Randomised backtracking for boundary cells
    # ------------------------------------------------------------------

    def _sample_boundary(
        self,
        boundary: List[Cell],
        constraints: List[Constraint],
        min_mines: int,
        max_mines: int,
    ) -> Optional[Assignment]:
        """
        Find one consistent assignment for boundary cells via randomised
        backtracking with incremental forward-checking.

        Key optimisation: instead of re-scanning all constraints at every node
        (O(C × N) per node), we:
          1. Pre-index which constraint indices each cell participates in.
          2. Maintain running `mines_count[i]` and `unset_count[i]` per
             constraint, updated in O(degree) on assign/unassign.
          3. On each assignment, only check the ~1-4 affected constraints.

        At the leaf (all cells assigned) every constraint's `unset_count` is 0,
        and the incremental forward checks guarantee `mines_count[i] == needed[i]`
        for all i, so no full consistency scan is required at the leaf.
        """
        order = boundary[:]
        random.shuffle(order)
        n = len(order)

        # Build cell → list of constraint indices it participates in
        cell_cidx: Dict[Cell, List[int]] = defaultdict(list)
        for i, (nbrs, _) in enumerate(constraints):
            for cell in nbrs:
                cell_cidx[cell].append(i)

        # Per-constraint running state
        mines_count: List[int] = [0] * len(constraints)
        unset_count: List[int] = [len(nbrs) for nbrs, _ in constraints]
        needed: List[int] = [req for _, req in constraints]

        # Flat assignment array (index aligns with `order`)
        values_arr: List[int] = [0] * n

        def _assign(idx: int, val: int) -> None:
            values_arr[idx] = val
            for ci in cell_cidx[order[idx]]:
                mines_count[ci] += val
                unset_count[ci] -= 1

        def _unassign(idx: int, val: int) -> None:
            values_arr[idx] = 0
            for ci in cell_cidx[order[idx]]:
                mines_count[ci] -= val
                unset_count[ci] += 1

        def _feasible(idx: int, val: int) -> bool:
            """Forward-check only the constraints containing order[idx]."""
            for ci in cell_cidx[order[idx]]:
                m = mines_count[ci] + val
                u = unset_count[ci] - 1
                req = needed[ci]
                if m > req or m + u < req:
                    return False
            return True

        nodes = [0]

        def backtrack(idx: int, placed: int) -> bool:
            if nodes[0] > self.max_bt_attempts:
                return False
            nodes[0] += 1

            if idx == n:
                # All constraints are exactly satisfied by induction from
                # the incremental forward checks (unset==0 ⟹ mines==needed).
                return min_mines <= placed <= max_mines

            vals = [0, 1] if random.random() < 0.5 else [1, 0]
            for val in vals:
                if val == 1 and placed >= max_mines:
                    continue
                # Prune: even placing all remaining as mines can't reach min
                if val == 0 and placed + (n - idx - 1) < min_mines:
                    continue

                if _feasible(idx, val):
                    _assign(idx, val)
                    if backtrack(idx + 1, placed + val):
                        return True
                    _unassign(idx, val)
            return False

        if backtrack(0, 0):
            return {order[i]: values_arr[i] for i in range(n)}
        return None
