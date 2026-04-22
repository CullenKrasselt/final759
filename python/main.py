"""
main.py — Monte Carlo Minesweeper demo.

Runs the solver autonomously on a configurable board.
Optionally prints each move with the estimated probability map.
"""

import argparse
import time

from game import MinesweeperGame, UNKNOWN
from solver import MonteCarloSolver


def run_game(
    rows: int = 9,
    cols: int = 9,
    mines: int = 10,
    n_samples: int = 500,
    verbose: bool = True,
) -> bool:
    """Play one game autonomously. Returns True if won."""
    game = MinesweeperGame(rows, cols, mines)
    solver = MonteCarloSolver(n_samples=n_samples)

    # First move: centre cell (pre-info, any choice is equally uninformed)
    first_r, first_c = rows // 2, cols // 2
    game.reveal(first_r, first_c)
    move_n = 1

    if verbose:
        print(f"Board: {rows}x{cols}, mines: {mines}, samples: {n_samples}")
        print(f"Move {move_n}: opening ({first_r},{first_c})\n")
        game.print_board()

    while not game.game_over:
        probs = solver.get_probabilities(game.board, rows, cols, mines)
        if not probs:
            break

        move = min(probs, key=probs.get)   # type: ignore[arg-type]
        r, c = move
        move_n += 1

        if verbose:
            print(f"\nMove {move_n}: ({r},{c})  p(mine)={probs[move]:.3f}")

        safe = game.reveal(r, c)

        if verbose:
            game.print_board()
            if not safe:
                print("  *** MINE HIT ***")

    result = "WON" if game.won else "LOST"
    if verbose:
        print(f"\n{'='*30}")
        print(f"Result: {result} in {move_n} moves")
        print("Final board (X = unrevealed mine):")
        game.print_board(reveal_all=True)

    return game.won


def benchmark(
    rows: int = 9,
    cols: int = 9,
    mines: int = 10,
    n_samples: int = 500,
    n_games: int = 100,
) -> None:
    """Play n_games silently and print win-rate statistics."""
    wins = 0
    t0 = time.perf_counter()
    for i in range(n_games):
        won = run_game(rows, cols, mines, n_samples, verbose=False)
        wins += won
        print(f"\r  Games played: {i+1}/{n_games}  Wins: {wins}", end="", flush=True)
    elapsed = time.perf_counter() - t0
    print()
    print(f"\nBenchmark ({rows}x{cols}, {mines} mines, {n_samples} samples/game)")
    print(f"  Win rate : {wins}/{n_games}  ({100*wins/n_games:.1f}%)")
    print(f"  Total time: {elapsed:.1f}s  ({elapsed/n_games:.2f}s/game)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Monte Carlo Minesweeper solver")
    parser.add_argument("--rows",      type=int, default=9,    help="Board rows")
    parser.add_argument("--cols",      type=int, default=9,    help="Board columns")
    parser.add_argument("--mines",     type=int, default=10,   help="Number of mines")
    parser.add_argument("--samples",   type=int, default=500,  help="MC samples per move")
    parser.add_argument("--games",     type=int, default=1,    help="Games to play (>1 = benchmark)")
    parser.add_argument("--quiet",     action="store_true",    help="Suppress per-move output")

    args = parser.parse_args()

    if args.games == 1:
        run_game(args.rows, args.cols, args.mines, args.samples, verbose=not args.quiet)
    else:
        benchmark(args.rows, args.cols, args.mines, args.samples, args.games)


if __name__ == "__main__":
    main()
