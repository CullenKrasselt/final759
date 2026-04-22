/**
 * solver.cpp — Monte Carlo Minesweeper solver implementation.
 */
#include "solver.hpp"
#include "game.hpp"   // for CELL_UNKNOWN / CELL_MINE constants

#include <algorithm>
#include <functional>
#include <numeric>
#include <unordered_set>

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

MonteCarloSolver::MonteCarloSolver(int n_samples, int max_bt_attempts)
    : n_samples(n_samples), max_bt_attempts(max_bt_attempts) {}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

std::vector<double>
MonteCarloSolver::get_probabilities(const std::vector<int>& board,
                                     int rows, int cols, int total_mines) {
    const int N = rows * cols;
    std::vector<double> probs(N, -1.0);

    auto unknown = _unknown_cells(board);
    if (unknown.empty()) return probs;

    int confirmed_mines = (int)std::count(board.begin(), board.end(), CELL_MINE);
    int remaining       = total_mines - confirmed_mines;

    auto constraints        = _extract_constraints(board, rows, cols);
    auto [boundary, interior] = _partition(unknown, constraints);

    // Fast path: no constraints (all cells are interior)
    if (constraints.empty()) {
        double p = (double)remaining / (double)unknown.size();
        for (int cell : unknown) probs[cell] = p;
        return probs;
    }

    int min_b = std::max(0, remaining - (int)interior.size());
    int max_b = std::min(remaining, (int)boundary.size());

    std::vector<double> mine_hits(N, 0.0);
    int valid = 0;

    // -----------------------------------------------------------------------
    // Sample loop — parallelise here with OpenMP in the next phase:
    //
    //   #pragma omp parallel for schedule(dynamic) reduction(+:valid)
    //   for (int s = 0; s < n_samples; ++s) {
    //       std::mt19937 local_rng(_rng() ^ (unsigned)omp_get_thread_num());
    //       ... (use local_rng instead of _rng, accumulate into thread-local
    //            mine_hits, then reduce)
    //   }
    // -----------------------------------------------------------------------
    for (int s = 0; s < n_samples; ++s) {
        auto b_result = _sample_boundary(boundary, constraints, min_b, max_b, _rng);
        if (!b_result) continue;

        int b_mines = std::accumulate(b_result->begin(), b_result->end(), 0);
        int i_mines = remaining - b_mines;
        if (i_mines < 0 || i_mines > (int)interior.size()) continue;

        // Sample interior cells uniformly
        std::vector<int> interior_copy = interior;
        std::shuffle(interior_copy.begin(), interior_copy.end(), _rng);

        ++valid;
        for (int j = 0; j < (int)boundary.size(); ++j)
            mine_hits[boundary[j]] += (*b_result)[j];
        for (int j = 0; j < i_mines; ++j)
            mine_hits[interior_copy[j]] += 1.0;
    }

    if (valid == 0) {
        double p = (double)remaining / (double)unknown.size();
        for (int cell : unknown) probs[cell] = p;
        return probs;
    }

    for (int cell : unknown)
        probs[cell] = mine_hits[cell] / (double)valid;
    return probs;
}

int MonteCarloSolver::best_move(const std::vector<int>& board,
                                 int rows, int cols, int total_mines) {
    auto probs = get_probabilities(board, rows, cols, total_mines);
    int best = -1;
    double best_p = 2.0;
    for (int i = 0; i < (int)probs.size(); ++i) {
        if (probs[i] >= 0.0 && probs[i] < best_p) {
            best_p = probs[i];
            best   = i;
        }
    }
    return best;
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

std::vector<int>
MonteCarloSolver::_unknown_cells(const std::vector<int>& board) const {
    std::vector<int> result;
    for (int i = 0; i < (int)board.size(); ++i)
        if (board[i] == CELL_UNKNOWN) result.push_back(i);
    return result;
}

std::vector<Constraint>
MonteCarloSolver::_extract_constraints(const std::vector<int>& board,
                                        int rows, int cols) const {
    std::vector<Constraint> constraints;
    auto for_neighbors = [&](int r, int c, auto fn) {
        for (int dr = -1; dr <= 1; ++dr)
            for (int dc = -1; dc <= 1; ++dc) {
                if (dr == 0 && dc == 0) continue;
                int nr = r + dr, nc = c + dc;
                if (nr >= 0 && nr < rows && nc >= 0 && nc < cols)
                    fn(nr * cols + nc);
            }
    };

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            int val = board[r * cols + c];
            if (val < 0) continue;   // UNKNOWN or MINE — not a clue cell

            Constraint cst;
            int mine_nbrs = 0;
            for_neighbors(r, c, [&](int ni) {
                if (board[ni] == CELL_UNKNOWN) cst.neighbors.push_back(ni);
                else if (board[ni] == CELL_MINE) ++mine_nbrs;
            });
            cst.needed = val - mine_nbrs;
            if (!cst.neighbors.empty())
                constraints.push_back(std::move(cst));
        }
    }
    return constraints;
}

std::pair<std::vector<int>, std::vector<int>>
MonteCarloSolver::_partition(const std::vector<int>& unknown_cells,
                               const std::vector<Constraint>& constraints) const {
    std::unordered_set<int> boundary_set;
    for (const auto& cst : constraints)
        for (int cell : cst.neighbors)
            boundary_set.insert(cell);

    std::vector<int> boundary, interior;
    for (int cell : unknown_cells) {
        if (boundary_set.count(cell)) boundary.push_back(cell);
        else                          interior.push_back(cell);
    }
    return {boundary, interior};
}

std::optional<std::vector<int>>
MonteCarloSolver::_sample_boundary(const std::vector<int>& boundary,
                                    const std::vector<Constraint>& constraints,
                                    int min_mines, int max_mines,
                                    std::mt19937& rng) const {
    const int n = (int)boundary.size();
    if (n == 0) return std::vector<int>{};

    // Randomise processing order
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::shuffle(order.begin(), order.end(), rng);

    // Build cell → constraint-index list (only for boundary cells)
    std::unordered_map<int, std::vector<int>> cell_cidx;
    for (int ci = 0; ci < (int)constraints.size(); ++ci)
        for (int cell : constraints[ci].neighbors)
            cell_cidx[cell].push_back(ci);

    // Incremental constraint state
    const int nc = (int)constraints.size();
    std::vector<int> mines_count(nc, 0);
    std::vector<int> unset_count(nc);
    std::vector<int> needed(nc);
    for (int i = 0; i < nc; ++i) {
        unset_count[i] = (int)constraints[i].neighbors.size();
        needed[i]      = constraints[i].needed;
    }

    std::vector<int> values_arr(n, 0);   // current assignment (index = position in order)
    int nodes = 0;

    // Assign order[pos] = val and update running counts
    auto assign = [&](int pos, int val) {
        values_arr[pos] = val;
        for (int ci : cell_cidx[boundary[order[pos]]]) {
            mines_count[ci] += val;
            --unset_count[ci];
        }
    };
    auto unassign = [&](int pos, int val) {
        values_arr[pos] = 0;
        for (int ci : cell_cidx[boundary[order[pos]]]) {
            mines_count[ci] -= val;
            ++unset_count[ci];
        }
    };

    // Forward-check: only examine constraints containing the current cell
    auto feasible = [&](int pos, int val) -> bool {
        for (int ci : cell_cidx[boundary[order[pos]]]) {
            int m = mines_count[ci] + val;
            int u = unset_count[ci] - 1;
            if (m > needed[ci] || m + u < needed[ci]) return false;
        }
        return true;
    };

    // Randomised backtracking
    std::function<bool(int, int)> backtrack = [&](int idx, int placed) -> bool {
        if (++nodes > max_bt_attempts) return false;
        if (idx == n) return min_mines <= placed && placed <= max_mines;

        int vals[2] = {0, 1};
        if (rng() & 1u) std::swap(vals[0], vals[1]);

        for (int val : vals) {
            if (val == 1 && placed >= max_mines) continue;
            if (val == 0 && placed + (n - idx - 1) < min_mines) continue;
            if (feasible(idx, val)) {
                assign(idx, val);
                if (backtrack(idx + 1, placed + val)) return true;
                unassign(idx, val);
            }
        }
        return false;
    };

    if (!backtrack(0, 0)) return std::nullopt;

    // Map back: result[j] = value for boundary[j]
    std::vector<int> result(n);
    for (int i = 0; i < n; ++i)
        result[order[i]] = values_arr[i];
    return result;
}
