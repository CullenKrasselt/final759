/**
 * solver.hpp — Monte Carlo Minesweeper solver interface.
 *
 * Algorithm (per sample):
 *   1. Extract constraints from revealed number cells.
 *   2. Partition unknown cells into:
 *        boundary — adjacent to >= 1 revealed clue (constrained)
 *        interior — no revealed neighbours (only total mine count applies)
 *   3. Randomised backtracking with incremental forward-checking
 *      assigns 0/1 to each boundary cell.
 *   4. Remaining mines are distributed uniformly over interior cells.
 *   5. Per-cell mine hits are tallied across all valid samples.
 *
 * Mine probability ~= hits[cell] / valid_samples.
 * Best move = argmin of that map.
 *
 * The sample loop in get_probabilities is intentionally written as a plain
 * for-loop so that an OpenMP parallel-for + reduction can be dropped in
 * with minimal changes.
 */
#pragma once

#include <optional>
#include <random>
#include <unordered_map>
#include <utility>
#include <vector>

struct Constraint {
    std::vector<int> neighbors;  // flat board indices of unknown neighbours
    int needed;                   // mines still required among those neighbours
};

class MonteCarloSolver {
public:
    int n_samples;
    int max_bt_attempts;

    explicit MonteCarloSolver(int n_samples      = 500,
                              int max_bt_attempts = 2000);

    /**
     * Returns a vector of size rows*cols where:
     *   -1.0  => cell is not unknown
     *   [0,1] => estimated mine probability
     */
    std::vector<double> get_probabilities(const std::vector<int>& board,
                                          int rows, int cols,
                                          int total_mines);

    /**
     * Returns the flat board index of the safest unknown cell,
     * or -1 if no unknown cells remain.
     */
    int best_move(const std::vector<int>& board,
                  int rows, int cols,
                  int total_mines);

private:
    std::mt19937 _rng{std::random_device{}()};

    std::vector<int> _unknown_cells(const std::vector<int>& board) const;

    std::vector<Constraint> _extract_constraints(const std::vector<int>& board,
                                                  int rows, int cols) const;

    std::pair<std::vector<int>, std::vector<int>>
    _partition(const std::vector<int>& unknown_cells,
               const std::vector<Constraint>& constraints) const;

    /**
     * Randomised backtracking to find one consistent boundary assignment.
     * Returns a vector parallel to `boundary` with values 0/1,
     * or std::nullopt if no solution found within max_bt_attempts nodes.
     * Takes rng by reference so the caller controls seeding (important for
     * the future OpenMP version where each thread owns its own rng).
     */
    std::optional<std::vector<int>>
    _sample_boundary(const std::vector<int>& boundary,
                     const std::vector<Constraint>& constraints,
                     int min_mines, int max_mines,
                     std::mt19937& rng) const;
};
