/**
 * main_cuda.cpp — Monte Carlo Minesweeper solver demo & benchmark (CUDA version).
 *
 * Identical CLI interface to main.cpp so the two binaries can be benchmarked
 * side-by-side with the same arguments:
 *
 *   ./minesweeper_mc   --games 100 --quiet --samples 500
 *   ./minesweeper_cuda --games 100 --quiet --samples 500
 *
 * Usage:
 *   ./minesweeper_cuda [--rows R] [--cols C] [--mines M]
 *                      [--samples S] [--games G] [--quiet]
 */
#include "game.hpp"
#include "solver_cuda.cuh"

#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>

// ---------------------------------------------------------------------------
// Argument parsing (identical to main.cpp)
// ---------------------------------------------------------------------------

static int get_int_arg(int argc, char** argv, const char* name, int def) {
    for (int i = 1; i < argc - 1; ++i)
        if (std::strcmp(argv[i], name) == 0)
            return std::stoi(argv[i + 1]);
    return def;
}

static bool has_flag(int argc, char** argv, const char* name) {
    for (int i = 1; i < argc; ++i)
        if (std::strcmp(argv[i], name) == 0) return true;
    return false;
}

// ---------------------------------------------------------------------------
// Single game (mirrors main.cpp::run_game, uses cuda_best_move)
// ---------------------------------------------------------------------------

static bool run_game(int rows, int cols, int mines, int n_samples,
                     bool verbose) {
    MinesweeperGame game(rows, cols, mines);

    int first_r = rows / 2, first_c = cols / 2;
    game.reveal(first_r, first_c);
    int move_n = 1;

    if (verbose) {
        std::cout << "[CUDA] Board: " << rows << "x" << cols
                  << ", mines: " << mines
                  << ", samples: " << n_samples << "\n";
        std::cout << "Move 1: opening (" << first_r << "," << first_c << ")\n";
        game.print_board();
    }

    while (!game.game_over) {
        auto probs = cuda_get_probabilities(game.board, rows, cols, mines,
                                             n_samples);
        int move = -1;
        double best_p = 2.0;
        for (int i = 0; i < (int)probs.size(); ++i) {
            if (probs[i] >= 0.0 && probs[i] < best_p) {
                best_p = probs[i];
                move   = i;
            }
        }
        if (move < 0) break;

        int r = move / cols, c = move % cols;
        ++move_n;

        if (verbose) {
            std::cout << "\nMove " << move_n
                      << ": (" << r << "," << c << ")"
                      << "  p(mine)=" << std::fixed << std::setprecision(3)
                      << probs[move] << "\n";
        }

        bool safe = game.reveal(r, c);

        if (verbose) {
            game.print_board();
            if (!safe) std::cout << "  *** MINE HIT ***\n";
        }
    }

    if (verbose) {
        std::cout << "\n" << std::string(30, '=') << "\n";
        std::cout << "Result: " << (game.won ? "WON" : "LOST")
                  << " in " << move_n << " moves\n";
        std::cout << "Final board (X = unrevealed mine):\n";
        game.print_board(true);
    }

    return game.won;
}

// ---------------------------------------------------------------------------
// Benchmark
// ---------------------------------------------------------------------------

static void benchmark(int rows, int cols, int mines, int n_samples,
                      int n_games) {
    int wins = 0;
    auto t0 = std::chrono::steady_clock::now();

    for (int g = 0; g < n_games; ++g) {
        wins += run_game(rows, cols, mines, n_samples, false) ? 1 : 0;
        std::cout << "\r  [CUDA] Games: " << (g + 1) << "/" << n_games
                  << "  Wins: " << wins << std::flush;
    }

    auto t1 = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "\n\n[CUDA] Benchmark (" << rows << "x" << cols
              << ", " << mines << " mines, " << n_samples << " samples/move)\n";
    std::cout << "  Win rate  : " << wins << "/" << n_games
              << "  (" << std::fixed << std::setprecision(1)
              << 100.0 * wins / n_games << "%)\n";
    std::cout << "  Total time: " << std::setprecision(2) << elapsed << "s"
              << "  (" << elapsed / n_games << "s/game)\n";
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    int rows    = get_int_arg(argc, argv, "--rows",    9);
    int cols    = get_int_arg(argc, argv, "--cols",    9);
    int mines   = get_int_arg(argc, argv, "--mines",   10);
    int samples = get_int_arg(argc, argv, "--samples", 500);
    int games   = get_int_arg(argc, argv, "--games",   1);
    bool quiet  = has_flag(argc, argv, "--quiet");

    if (games == 1)
        run_game(rows, cols, mines, samples, !quiet);
    else
        benchmark(rows, cols, mines, samples, games);

    return 0;
}
