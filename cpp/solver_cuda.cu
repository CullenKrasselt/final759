/**
 * solver_cuda.cu — CUDA Monte Carlo Minesweeper solver.
 *
 * Host side:
 *   - Reuses constraint extraction and boundary/interior partitioning
 *     verbatim from the sequential solver (CPU only, runs once per move).
 *   - Flattens all STL structures into plain arrays for device transfer.
 *   - Launches one thread per sample; collects per-cell mine-hit tallies
 *     via atomics into a device accumulator, then normalises on the host.
 *
 * Device side (kernel):
 *   - Each thread seeds its own cuRAND XORWOW state from a base seed XOR'd
 *     with the thread's global ID — identical seeding strategy to the
 *     OpenMP version (local_rng(_rng() ^ thread_id)).
 *   - Iterative backtracking replaces the recursive _sample_boundary:
 *     a fixed-size per-thread stack (MAX_BOUNDARY_CELLS frames) lives in
 *     local (register-spill) memory; no heap allocation, no recursion.
 *   - Interior cells are shuffled with a simple Fisher-Yates using cuRAND.
 *   - Atomic adds to a global int accumulator (scaled by SCALE) avoid the
 *     need for a shared-memory reduction tree while keeping precision.
 */

#include "solver_cuda.cuh"
#include "solver.hpp"   // Constraint, MonteCarloSolver helpers (host only)
#include "game.hpp"     // CELL_UNKNOWN, CELL_MINE

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// ---------------------------------------------------------------------------
// Compile-time limits
// ---------------------------------------------------------------------------

// Maximum boundary cells we support on the GPU.
// Expert board (16x30, 99 mines) typically has ~80-120 boundary cells.
static constexpr int MAX_BOUNDARY_CELLS   = 256;

// Maximum constraints per board move.
static constexpr int MAX_CONSTRAINTS      = 512;

// Maximum neighbours stored per constraint (8 at most on a grid).
static constexpr int MAX_NBR_PER_CST      = 8;

// Maximum cells a single boundary cell can participate in (its degree).
static constexpr int MAX_CELL_DEGREE      = 8;

// Fixed-point scale for atomic accumulation of float hits.
static constexpr int SCALE                = 1000;

// ---------------------------------------------------------------------------
// Flat device-side data structures (no STL, no pointers-to-pointers)
// ---------------------------------------------------------------------------

struct DevConstraint {
    int neighbors[MAX_NBR_PER_CST];
    int n_neighbors;
    int needed;
};

// Per boundary-cell: which constraint indices touch it (for forward-checking)
struct DevCellDeps {
    int cidx[MAX_CELL_DEGREE];
    int n_deps;
};

// ---------------------------------------------------------------------------
// Stack frame for iterative backtracking
// ---------------------------------------------------------------------------

struct Frame {
    int idx;        // boundary cell index being decided (position in order[])
    int placed;     // mines placed so far before this frame
    int next_val;   // next value to try: 0 or 1 (2 = both tried, backtrack)
};

// ---------------------------------------------------------------------------
// Device kernel
// ---------------------------------------------------------------------------

__global__ void mc_kernel(
    // Boundary cells (flat board indices, shuffled per-sample on device)
    const int*          d_boundary,
    int                 n_boundary,
    // Interior cells (flat board indices)
    const int*          d_interior,
    int                 n_interior,
    // Constraints
    const DevConstraint* d_constraints,
    int                 n_constraints,
    // Per-cell dependency lists (index into d_constraints)
    const DevCellDeps*  d_cell_deps,   // indexed by position in d_boundary
    // Mine count bounds
    int                 min_mines,
    int                 max_mines,
    int                 remaining,
    // Sampling config
    int                 n_samples,
    int                 max_bt_attempts,
    unsigned long long  base_seed,
    // Output: scaled integer mine-hit counts, indexed by flat board index
    int*                d_hits,
    int                 board_size,
    // Output: number of valid samples accepted
    int*                d_valid
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_samples) return;

    // -----------------------------------------------------------------------
    // Per-thread RNG (XORWOW) — seeded uniquely per thread
    // -----------------------------------------------------------------------
    curandStateXORWOW_t rng;
    curand_init(base_seed ^ (unsigned long long)tid, tid, 0, &rng);

    // -----------------------------------------------------------------------
    // Local copies of per-sample state (registers / local memory)
    // -----------------------------------------------------------------------

    // Processing order for boundary cells (randomised each sample)
    int order[MAX_BOUNDARY_CELLS];
    for (int i = 0; i < n_boundary; ++i) order[i] = i;

    // Fisher-Yates shuffle of order[]
    for (int i = n_boundary - 1; i > 0; --i) {
        unsigned r = curand(&rng) % (unsigned)(i + 1);
        int tmp = order[i]; order[i] = order[r]; order[r] = tmp;
    }

    // Incremental constraint state
    int mines_count[MAX_CONSTRAINTS];
    int unset_count[MAX_CONSTRAINTS];
    int needed_local[MAX_CONSTRAINTS];
    for (int ci = 0; ci < n_constraints; ++ci) {
        mines_count[ci] = 0;
        unset_count[ci] = d_constraints[ci].n_neighbors;
        needed_local[ci] = d_constraints[ci].needed;
    }

    // Assignment array: values_arr[pos] = 0 or 1 for boundary cell order[pos]
    int values_arr[MAX_BOUNDARY_CELLS];
    for (int i = 0; i < n_boundary; ++i) values_arr[i] = 0;

    // -----------------------------------------------------------------------
    // Iterative backtracking (replaces recursive _sample_boundary)
    // -----------------------------------------------------------------------

    Frame stack[MAX_BOUNDARY_CELLS];
    int  stack_top = 0;
    int  nodes     = 0;
    bool found     = false;

    // Push initial frame
    stack[0] = {0, 0, -1};  // next_val = -1 means "not started yet"
    stack_top = 1;

    while (stack_top > 0 && nodes <= max_bt_attempts) {
        Frame& f = stack[stack_top - 1];
        int idx    = f.idx;
        int placed = f.placed;

        // ---- Terminal: all boundary cells assigned ----
        if (idx == n_boundary) {
            if (placed >= min_mines && placed <= max_mines) {
                found = true;
            }
            // Either way, pop and backtrack
            --stack_top;
            // Unassign the previous cell (the one that led us here)
            if (stack_top > 0) {
                Frame& parent = stack[stack_top - 1];
                int ppos = parent.idx - 1;  // position just assigned
                if (ppos >= 0) {
                    int pval = values_arr[ppos];
                    values_arr[ppos] = 0;
                    int bcell_pos = order[ppos];
                    for (int d = 0; d < d_cell_deps[bcell_pos].n_deps; ++d) {
                        int ci = d_cell_deps[bcell_pos].cidx[d];
                        mines_count[ci] -= pval;
                        ++unset_count[ci];
                    }
                }
            }
            if (found) break;
            continue;
        }

        ++nodes;

        // ---- Choose next value to try ----
        // next_val cycles: -1 (init) -> first try -> second try -> done
        // We randomise first-try with cuRAND (50/50 start with 0 or 1)
        int val;
        bool exhausted = false;

        if (f.next_val == -1) {
            // First visit: pick a random starting value
            f.next_val = (int)(curand(&rng) & 1u);
            val = f.next_val;
        } else if (f.next_val == 0) {
            val = 1;
            f.next_val = 2;
        } else if (f.next_val == 1) {
            val = 0;
            f.next_val = 2;
        } else {
            exhausted = true;
        }

        if (exhausted) {
            // Both values tried — backtrack
            --stack_top;
            // Undo the assignment for this idx (already unassigned by child)
            // Actually we need to undo what was assigned when we first pushed
            // this frame's predecessor. Handled by the parent's unassign step.
            // But we also need to unassign ourselves if we made any assignment.
            // Since we track values_arr separately, undo current pos if needed.
            if (stack_top > 0) {
                // The assignment for (idx-1) was already unassigned when we
                // popped the child frame above. Here idx is the *current* frame's
                // cell, so unassign idx-1 (what the parent assigned to arrive here).
                // Actually: when we push a child for idx+1, we assign idx first.
                // So on backtrack from idx, unassign idx.
                int ppos = idx - 1;
                if (ppos >= 0 && stack_top > 0) {
                    // unassign ppos — but this was handled in the child pop above.
                    // Nothing extra needed here; values_arr[ppos] reset in child pop.
                }
            }
            continue;
        }

        // ---- Pruning ----
        // Can't place more mines than max_mines
        if (val == 1 && placed >= max_mines) {
            // Try the other value next iteration (don't advance next_val twice)
            // Flip: if we just set next_val to 2, we need to still try the other
            // Simplest: just mark next_val as exhausted on this val, loop again
            // We handle this by continuing without advancing idx.
            // Re-enter loop will pick the other val via next_val state.
            // But next_val is already advanced — we need to "un-advance" or just skip.
            // Simplest fix: just treat it as if feasibility failed (loop continues).
            continue;
        }
        if (val == 0 && (placed + (n_boundary - idx - 1)) < min_mines) {
            continue;
        }

        // ---- Forward-check ----
        bool ok = true;
        int bcell_pos = order[idx];
        for (int d = 0; d < d_cell_deps[bcell_pos].n_deps; ++d) {
            int ci = d_cell_deps[bcell_pos].cidx[d];
            int m  = mines_count[ci] + val;
            int u  = unset_count[ci] - 1;
            if (m > needed_local[ci] || m + u < needed_local[ci]) {
                ok = false;
                break;
            }
        }
        if (!ok) continue;

        // ---- Assign and push child ----
        values_arr[idx] = val;
        for (int d = 0; d < d_cell_deps[bcell_pos].n_deps; ++d) {
            int ci = d_cell_deps[bcell_pos].cidx[d];
            mines_count[ci] += val;
            --unset_count[ci];
        }

        // Push child frame for idx+1
        if (stack_top < MAX_BOUNDARY_CELLS) {
            stack[stack_top++] = {idx + 1, placed + val, -1};
        }
    }

    if (!found) return;

    // -----------------------------------------------------------------------
    // Count boundary mines placed
    // -----------------------------------------------------------------------
    int b_mines = 0;
    for (int i = 0; i < n_boundary; ++i) b_mines += values_arr[i];

    int i_mines = remaining - b_mines;
    if (i_mines < 0 || i_mines > n_interior) return;

    // -----------------------------------------------------------------------
    // Sample interior cells: Fisher-Yates partial shuffle of indices 0..n_interior-1
    // -----------------------------------------------------------------------
    int interior_order[MAX_BOUNDARY_CELLS];  // interior <= board size, reuse limit
    // We only need to shuffle the first i_mines positions
    for (int i = 0; i < n_interior; ++i) interior_order[i] = i;
    for (int i = 0; i < i_mines; ++i) {
        unsigned r = i + curand(&rng) % (unsigned)(n_interior - i);
        int tmp = interior_order[i];
        interior_order[i] = interior_order[r];
        interior_order[r] = tmp;
    }

    // -----------------------------------------------------------------------
    // Accumulate hits atomically into global output
    // -----------------------------------------------------------------------
    atomicAdd(d_valid, 1);

    for (int j = 0; j < n_boundary; ++j) {
        if (values_arr[j]) {
            atomicAdd(&d_hits[d_boundary[order[j]]], SCALE);
        }
    }
    for (int j = 0; j < i_mines; ++j) {
        atomicAdd(&d_hits[d_interior[interior_order[j]]], SCALE);
    }
}

// ---------------------------------------------------------------------------
// CUDA error checking macro
// ---------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                  \
    do {                                                                   \
        cudaError_t _e = (call);                                           \
        if (_e != cudaSuccess) {                                           \
            throw std::runtime_error(std::string("CUDA error: ")          \
                + cudaGetErrorString(_e) + " at " __FILE__ ":"            \
                + std::to_string(__LINE__));                               \
        }                                                                  \
    } while (0)

// ---------------------------------------------------------------------------
// Host-side helpers (mirror MonteCarloSolver private methods)
// ---------------------------------------------------------------------------

static std::vector<int> host_unknown_cells(const std::vector<int>& board) {
    std::vector<int> result;
    for (int i = 0; i < (int)board.size(); ++i)
        if (board[i] == CELL_UNKNOWN) result.push_back(i);
    return result;
}

static std::vector<Constraint>
host_extract_constraints(const std::vector<int>& board, int rows, int cols) {
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
            if (val < 0) continue;
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

static std::pair<std::vector<int>, std::vector<int>>
host_partition(const std::vector<int>& unknown,
               const std::vector<Constraint>& constraints) {
    std::unordered_set<int> bset;
    for (const auto& c : constraints)
        for (int cell : c.neighbors) bset.insert(cell);
    std::vector<int> boundary, interior;
    for (int cell : unknown) {
        if (bset.count(cell)) boundary.push_back(cell);
        else                  interior.push_back(cell);
    }
    return {boundary, interior};
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

std::vector<double> cuda_get_probabilities(
    const std::vector<int>& board,
    int rows, int cols, int total_mines,
    int n_samples, int max_bt_attempts)
{
    const int N = rows * cols;
    std::vector<double> probs(N, -1.0);

    // ---- Host-side preprocessing ----
    auto unknown     = host_unknown_cells(board);
    if (unknown.empty()) return probs;

    int confirmed    = (int)std::count(board.begin(), board.end(), CELL_MINE);
    int remaining    = total_mines - confirmed;
    auto constraints = host_extract_constraints(board, rows, cols);
    auto [boundary, interior] = host_partition(unknown, constraints);

    // Fast path: no boundary cells
    if (constraints.empty() || boundary.empty()) {
        double p = remaining / (double)unknown.size();
        for (int cell : unknown) probs[cell] = p;
        return probs;
    }

    int n_b  = (int)boundary.size();
    int n_i  = (int)interior.size();
    int n_c  = (int)constraints.size();
    int min_b = std::max(0, remaining - n_i);
    int max_b = std::min(remaining, n_b);

    if (n_b > MAX_BOUNDARY_CELLS || n_c > MAX_CONSTRAINTS) {
        // Fallback: uniform probability
        double p = remaining / (double)unknown.size();
        for (int cell : unknown) probs[cell] = p;
        return probs;
    }

    // ---- Flatten constraints for device ----
    std::vector<DevConstraint> h_constraints(n_c);
    for (int ci = 0; ci < n_c; ++ci) {
        h_constraints[ci].n_neighbors = (int)constraints[ci].neighbors.size();
        h_constraints[ci].needed      = constraints[ci].needed;
        for (int j = 0; j < h_constraints[ci].n_neighbors; ++j)
            h_constraints[ci].neighbors[j] = constraints[ci].neighbors[j];
    }

    // Build cell -> constraint index map (keyed by position in boundary[])
    // boundary_pos_map: flat board index -> position in boundary[]
    std::unordered_map<int, int> boundary_pos_map;
    for (int i = 0; i < n_b; ++i) boundary_pos_map[boundary[i]] = i;

    std::vector<DevCellDeps> h_cell_deps(n_b);
    for (int i = 0; i < n_b; ++i) h_cell_deps[i].n_deps = 0;

    for (int ci = 0; ci < n_c; ++ci) {
        for (int cell : constraints[ci].neighbors) {
            auto it = boundary_pos_map.find(cell);
            if (it == boundary_pos_map.end()) continue;
            int pos = it->second;
            int& nd = h_cell_deps[pos].n_deps;
            if (nd < MAX_CELL_DEGREE)
                h_cell_deps[pos].cidx[nd++] = ci;
        }
    }

    // ---- Allocate and transfer device memory ----
    int *d_boundary, *d_interior, *d_hits, *d_valid;
    DevConstraint* d_constraints;
    DevCellDeps*   d_cell_deps;

    CUDA_CHECK(cudaMalloc(&d_boundary,    n_b * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_interior,    n_i * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_constraints, n_c * sizeof(DevConstraint)));
    CUDA_CHECK(cudaMalloc(&d_cell_deps,   n_b * sizeof(DevCellDeps)));
    CUDA_CHECK(cudaMalloc(&d_hits,        N   * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_valid,             sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_boundary,    boundary.data(),        n_b * sizeof(int),            cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_interior,    interior.data(),        n_i * sizeof(int),            cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_constraints, h_constraints.data(),   n_c * sizeof(DevConstraint),  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cell_deps,   h_cell_deps.data(),     n_b * sizeof(DevCellDeps),    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_hits,  0, N * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_valid, 0,     sizeof(int)));

    // ---- Launch kernel ----
    int threads = 256;
    int blocks  = (n_samples + threads - 1) / threads;

    unsigned long long seed =
        (unsigned long long)std::random_device{}() << 32 |
        (unsigned long long)std::random_device{}();

    mc_kernel<<<blocks, threads>>>(
        d_boundary, n_b,
        d_interior, n_i,
        d_constraints, n_c,
        d_cell_deps,
        min_b, max_b, remaining,
        n_samples, max_bt_attempts,
        seed,
        d_hits, N, d_valid
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---- Copy results back ----
    std::vector<int> h_hits(N);
    int h_valid = 0;
    CUDA_CHECK(cudaMemcpy(h_hits.data(), d_hits,  N * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_valid,      d_valid,     sizeof(int), cudaMemcpyDeviceToHost));

    // ---- Free device memory ----
    cudaFree(d_boundary);
    cudaFree(d_interior);
    cudaFree(d_constraints);
    cudaFree(d_cell_deps);
    cudaFree(d_hits);
    cudaFree(d_valid);

    // ---- Normalise ----
    if (h_valid == 0) {
        double p = remaining / (double)unknown.size();
        for (int cell : unknown) probs[cell] = p;
        return probs;
    }

    double denom = (double)h_valid * SCALE;
    for (int cell : unknown)
        probs[cell] = h_hits[cell] / denom;

    return probs;
}

int cuda_best_move(
    const std::vector<int>& board,
    int rows, int cols, int total_mines,
    int n_samples, int max_bt_attempts)
{
    auto probs = cuda_get_probabilities(board, rows, cols, total_mines,
                                        n_samples, max_bt_attempts);
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
