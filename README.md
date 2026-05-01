# Minesweeper Monte Carlo Solver
A base implementation of a Monte Carlo mine-probability estimator for Minesweeper.
## Files
| File | Purpose |
|------|---------|
| `game.py` | `MinesweeperGame` — programmatic game engine |
| `solver.py` | `MonteCarloSolver` — probability estimator |
| `main.py` | CLI demo / benchmark runner |
| `cpp/game.hpp` | C++ header-only game engine |
| `cpp/solver.hpp` / `cpp/solver.cpp` | C++ sequential & OpenMP solver |
| `cpp/main.cpp` | C++ CLI demo / benchmark runner |
| `cuda/solver_cuda.cuh` / `cuda/solver_cuda.cu` | CUDA GPU solver |
| `cuda/main_cuda.cpp` | CUDA CLI demo / benchmark runner |

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

```bash
# C++ build (sequential or OpenMP)
cmake -S cpp/ -B cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build
./cpp/build/minesweeper_mc --games 100 --quiet --samples 500

# CUDA build
cmake -S cuda/ -B cuda/build -DCMAKE_BUILD_TYPE=Release
cmake --build cuda/build
./cuda/build/minesweeper_cuda --games 100 --quiet --samples 500
```

## Requirements
Python ≥ 3.10. No external libraries required.
C++ build requires CMake ≥ 3.16 and a C++17 compiler.
CUDA build additionally requires the CUDA toolkit and an NVIDIA GPU.

## Benchmark Results

All benchmarks run on the UW–Madison Euler cluster (SLURM instruction partition).
Sequential and OpenMP runs: 1 CPU core / 8 CPU cores respectively.
CUDA run: 1 NVIDIA GPU, CUDA 13.0.

### Win Rate — 500 samples/move, 100 games

| Board | Sequential | OpenMP (8 threads) | CUDA |
|-------|:----------:|:------------------:|:----:|
| Beginner (9×9, 10 mines) | 94% | 95% | 93% |
| Intermediate (16×16, 40 mines) | 64% | 76% | 62% |
| Expert (16×30, 99 mines) | 1% | 2% | 0% |

Win rates are consistent across implementations within normal statistical variance,
confirming correctness of all three parallel versions. Expert board win rates are
low at 500 samples due to the difficulty of the constraint satisfaction problem
at that scale.

### Time per Game — 500 samples/move

| Board | Sequential | OpenMP (8 threads) | CUDA | OMP Speedup | CUDA Speedup |
|-------|:----------:|:------------------:|:----:|:-----------:|:------------:|
| Beginner (9×9, 10 mines) | 0.09s | 0.03s | 0.05s | 3.0× | 1.8× |
| Intermediate (16×16, 40 mines) | 3.46s | 0.81s | 1.42s | 4.3× | 2.4× |
| Expert (16×30, 99 mines) | 2.89s | 0.63s | 0.53s | 4.6× | 5.5× |

### Throughput Scaling — Beginner Board (9×9, 10 mines, 50 games)

| Samples/move | Sequential (s/game) | OpenMP 8t (s/game) | CUDA (s/game) |
|:------------:|:-------------------:|:------------------:|:-------------:|
| 100 | 0.02 | 0.12 | 0.03 |
| 250 | 0.05 | 0.01 | 0.05 |
| 500 | 0.11 | 0.05 | 0.05 |
| 1000 | 0.23 | 0.09 | 0.05 |
| 2500 | 0.44 | 0.14 | 0.05 |
| 5000 | 0.89 | 0.31 | 0.05 |

CUDA time is nearly flat across all sample counts, demonstrating the GPU's ability
to absorb increased parallelism. OpenMP shows overhead at low sample counts (100
samples/move is slower than sequential) due to thread spawn cost exceeding the
work per sample at small N.

### Throughput Scaling — Expert Board (16×30, 99 mines, 20 games)

| Samples/move | Sequential (s/game) | OpenMP 8t (s/game) | CUDA (s/game) |
|:------------:|:-------------------:|:------------------:|:-------------:|
| 100 | 0.14 | 0.20 | 0.10 |
| 250 | 0.70 | 0.37 | 0.33 |
| 500 | 4.51 | 0.50 | 0.57 |
| 1000 | 13.70 | 2.28 | 0.88 |
| 2500 | — | 6.27 | 1.80 |
| 5000 | — | 31.98 | 3.46 |
| 10000 | — | — | 3.83 |

Sequential results for 2500+ samples/move exceeded the cluster time limit.
At high sample counts on the expert board, CUDA pulls ahead of OpenMP significantly,
with CUDA achieving ~8× speedup over OpenMP at 5000 samples/move.

### OpenMP Thread Scaling — Beginner Board (9×9, 500 samples, 50 games)

| Threads | Time (s/game) | Speedup vs 1 thread |
|:-------:|:-------------:|:-------------------:|
| 1 | 0.09 | 1.0× |
| 2 | 0.04 | 2.0× |
| 4 | 0.04 | 2.0× |
| 8 | 0.10 | 0.9× |

Speedup plateaus at 2–4 threads and degrades at 8 threads due to `schedule(dynamic)`
overhead dominating when individual sample tasks are short. This is a known
limitation of dynamic scheduling at fine task granularity.
