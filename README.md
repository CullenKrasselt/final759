# Minesweeper Monte Carlo Solver

A base implementation of a Monte Carlo mine-probability estimator for Minesweeper.

## Files

| File | Purpose |
|------|---------|
| `game.py` | `MinesweeperGame` — programmatic game engine |
| `solver.py` | `MonteCarloSolver` — probability estimator |
| `main.py` | CLI demo / benchmark runner |

## Algorithm

For each of N samples the solver:

1. **Extracts constraints** — each revealed number `k` means exactly `k` of its
   unknown neighbours are mines.
2. **Partitions unknown cells** into:
   - *Boundary* — adjacent to ≥1 revealed clue (constrained).
   - *Interior* — no revealed neighbours (only global mine count applies).
3. **Randomised backtracking** assigns 0/1 to each boundary cell in a random
   order, pruning branches that violate any constraint (forward checking).
4. **Uniform random sampling** distributes the remaining mines over interior cells.
5. **Tallies** the per-cell mine hit across all valid samples.

Mine probability ≈ `hits[cell] / valid_samples`.  
Best move = `argmin` of that probability map.

## Usage

```bash
# Play one game with verbose output
python main.py

# Expert board (16x30, 99 mines), 1000 samples/move
python main.py --rows 16 --cols 30 --mines 99 --samples 1000

# Benchmark: 100 beginner games, quiet
python main.py --games 100 --quiet

# Use the solver in your own code
from game import MinesweeperGame
from solver import MonteCarloSolver

game = MinesweeperGame(9, 9, 10)
solver = MonteCarloSolver(n_samples=500)

game.reveal(4, 4)  # first move
while not game.game_over:
    r, c = solver.best_move(game.board, game.rows, game.cols, game.total_mines)
    game.reveal(r, c)
```

## Requirements

Python ≥ 3.10. No external libraries required.
