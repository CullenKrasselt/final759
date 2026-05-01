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
 *     with the thread's global ID.
 *   - Iterative backtracking replaces the recursive _sample_boundary.
 *     Uses explicit assignment[] and tried[] arrays as the stack state;
 *     depth is the only stack pointer needed.
 *   - Interior cells are shuffled with Fisher-Yates using cuRAND.
 *   - Atomic adds to a global int accumulator (scaled by SCALE).
 */

#include "solver_cuda.cuh"
#include "solver.hpp"
#include "game.hpp"

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
static constexpr int MAX_BOUNDARY_CELLS = 256;
static constexpr int MAX_CONSTRAINTS    = 512;
static constexpr int MAX_NBR_PER_CST   = 8;
static constexpr int MAX_CELL_DEGREE   = 8;
static constexpr int SCALE             = 1000;
static constexpr int MAX_INTERIOR_CELLS = 1024;

// ---------------------------------------------------------------------------
// Flat device-side data structures
// ---------------------------------------------------------------------------
struct DevConstraint {
    int neighbors[MAX_NBR_PER_CST];
    int n_neighbors;
    int needed;
};

struct DevCellDeps {
    int cidx[MAX_CELL_DEGREE];
    int n_deps;
};

// ---------------------------------------------------------------------------
// Device kernel
// ---------------------------------------------------------------------------
__global__ void mc_kernel(
    const int*           d_boundary,
    int                  n_boundary,
    const int*           d_interior,
    int                  n_interior,
    const DevConstraint* d_constraints,
    int                  n_constraints,
    const DevCellDeps*   d_cell_deps,   // indexed by position in d_boundary
    int                  min_mines,
    int                  max_mines,
    int                  remaining,
    int                  n_samples,
    int                  max_bt_attempts,
    unsigned long long   base_seed,
    int*                 d_hits,
    int                  board_size,
    int*                 d_valid
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_samples) return;

    // -----------------------------------------------------------------------
    // Per-thread RNG
    // -----------------------------------------------------------------------
    curandStateXORWOW_t rng;
    curand_init(base_seed ^ (unsigned long long)tid, tid, 0, &rng);

    // -----------------------------------------------------------------------
    // Randomise processing order for boundary cells
    // -----------------------------------------------------------------------
    int order[MAX_BOUNDARY_CELLS];
    for (int i = 0; i < n_boundary; ++i) order[i] = i;
    for (int i = n_boundary - 1; i > 0; --i) {
        unsigned r = curand(&rng) % (unsigned)(i + 1);
        int tmp = order[i]; order[i] = order[r]; order[r] = tmp;
    }

    // -----------------------------------------------------------------------
    // Incremental constraint state
    // -----------------------------------------------------------------------
    int mines_count[MAX_CONSTRAINTS];
    int unset_count[MAX_CONSTRAINTS];
    int needed_local[MAX_CONSTRAINTS];
    for (int ci = 0; ci < n_constraints; ++ci) {
        mines_count[ci]  = 0;
        unset_count[ci]  = d_constraints[ci].n_neighbors;
        needed_local[ci] = d_constraints[ci].needed;
    }

    // -----------------------------------------------------------------------
    // Iterative backtracking
    //
    // assignment[d] = value (0 or 1) assigned at depth d (-1 = unassigned)
    // tried[d]      = number of values already attempted at depth d (0,1,2)
    // first_val[d]  = the first value tried at depth d (so we know the other)
    // -----------------------------------------------------------------------
    int assignment[MAX_BOUNDARY_CELLS];
    int tried[MAX_BOUNDARY_CELLS];
    int first_val[MAX_BOUNDARY_CELLS];
    for (int i = 0; i < n_boundary; ++i) {
        assignment[i] = -1;
        tried[i]      = 0;
        first_val[i]  = -1;
    }

    int  depth = 0;
    int  nodes = 0;
    bool found = false;

    while (depth >= 0 && nodes <= max_bt_attempts) {

        // ---- Terminal: all boundary cells assigned ----
        if (depth == n_boundary) {
            int placed = 0;
            for (int i = 0; i < n_boundary; ++i) placed += assignment[i];
            if (placed >= min_mines && placed <= max_mines) {
                found = true;
                break;
            }
            // Mine count out of global bounds — backtrack
            depth--;
            if (depth < 0) break;
            {
                int pos = order[depth];
                int val = assignment[depth];
                assignment[depth] = -1;
                for (int d = 0; d < d_cell_deps[pos].n_deps; ++d) {
                    int ci = d_cell_deps[pos].cidx[d];
                    mines_count[ci] -= val;
                    ++unset_count[ci];
                }
            }
            continue;
        }

        ++nodes;

        // ---- Count mines placed so far ----
        int placed_so_far = 0;
        for (int i = 0; i < depth; ++i) placed_so_far += assignment[i];

        // ---- Try the next untried value at this depth ----
        bool advanced = false;
        while (tried[depth] < 2) {
            int val;
            if (tried[depth] == 0) {
                // First attempt: random starting value
                val = (int)(curand(&rng) & 1u);
                first_val[depth] = val;
            } else {
                // Second attempt: the other value
                val = 1 - first_val[depth];
            }
            tried[depth]++;

            // ---- Global mine count pruning ----
            if (val == 1 && placed_so_far >= max_mines) continue;
            if (val == 0 && (placed_so_far + (n_boundary - depth - 1)) < min_mines) continue;

            // ---- Forward-check constraints touching this cell ----
            int pos = order[depth];
            bool ok = true;
            for (int d = 0; d < d_cell_deps[pos].n_deps; ++d) {
                int ci = d_cell_deps[pos].cidx[d];
                int m  = mines_count[ci] + val;
                int u  = unset_count[ci] - 1;
                if (m > needed_local[ci] || m + u < needed_local[ci]) {
                    ok = false;
                    break;
                }
            }
            if (!ok) continue;

            // ---- Assign and descend ----
            assignment[depth] = val;
            for (int d = 0; d < d_cell_deps[pos].n_deps; ++d) {
                int ci = d_cell_deps[pos].cidx[d];
                mines_count[ci] += val;
                --unset_count[ci];
            }
            depth++;
            advanced = true;
            break;
        }

        if (!advanced) {
            // Both values exhausted at this depth — backtrack
            tried[depth]      = 0;
            first_val[depth]  = -1;
            assignment[depth] = -1;
            depth--;
            if (depth < 0) break;

            // Unassign the cell at the new (parent) depth
            int pos = order[depth];
            int val = assignment[depth];
            assignment[depth] = -1;
            for (int d = 0; d < d_cell_deps[pos].n_deps; ++d) {
                int ci = d_cell_deps[pos].cidx[d];
                mines_count[ci] -= val;
                ++unset_count[ci];
            }
        }
    }

    if (!found) return;

    // -----------------------------------------------------------------------
    // Count boundary mines and compute interior mine count
    // -----------------------------------------------------------------------
    int b_mines = 0;
    for (int i = 0; i < n_boundary; ++i) b_mines += assignment[i];

    int i_mines = remaining - b_mines;
    if (i_mines < 0 || i_mines > n_interior) return;
    if (n_interior > MAX_INTERIOR_CELLS) return;
    // -----------------------------------------------------------------------
    // Partial Fisher-Yates shuffle of interior indices
    // -----------------------------------------------------------------------
    int interior_order[MAX_INTERIOR_CELLS];
    for (int i = 0; i < n_interior; ++i) interior_order[i] = i;
    for (int i = 0; i < i_mines; ++i) {
        unsigned r = i + curand(&rng) % (unsigned)(n_interior - i);
        int tmp = interior_order[i];
        interior_order[i] = interior_order[r];
        interior_order[r] = tmp;
    }

    // -----------------------------------------------------------------------
    // Accumulate hits atomically
    // -----------------------------------------------------------------------
    atomicAdd(d_valid, 1);

    for (int j = 0; j < n_boundary; ++j) {
        if (assignment[j]) {
            // assignment[j] is the value for boundary cell order[j]
            atomicAdd(&d_hits[d_boundary[order[j]]], SCALE);
        }
    }
    for (int j = 0; j < i_mines; ++j) {
        atomicAdd(&d_hits[d_interior[interior_order[j]]], SCALE);
    }
}

// ---------------------------------------------------------------------------
// CUDA error checking
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                   \
    do {                                                                    \
        cudaError_t _e = (call);                                            \
        if (_e != cudaSuccess) {                                            \
            throw std::runtime_error(std::string("CUDA error: ")           \
                + cudaGetErrorString(_e) + " at " __FILE__ ":"             \
                + std::to_string(__LINE__));                                \
        }                                                                   \
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
                if (board[ni] == CELL_UNKNOWN)   cst.neighbors.push_back(ni);
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

    auto unknown  = host_unknown_cells(board);
    if (unknown.empty()) return probs;

    int confirmed = (int)std::count(board.begin(), board.end(), CELL_MINE);
    int remaining = total_mines - confirmed;
    auto constraints = host_extract_constraints(board, rows, cols);
    auto [boundary, interior] = host_partition(unknown, constraints);

    if (constraints.empty() || boundary.empty()) {
        double p = remaining / (double)unknown.size();
        for (int cell : unknown) probs[cell] = p;
        return probs;
    }

    int n_b = (int)boundary.size();
    int n_i = (int)interior.size();
    int n_c = (int)constraints.size();
    int min_b = std::max(0, remaining - n_i);
    int max_b = std::min(remaining, n_b);

    if (n_b > MAX_BOUNDARY_CELLS || n_c > MAX_CONSTRAINTS) {
        double p = remaining / (double)unknown.size();
        for (int cell : unknown) probs[cell] = p;
        return probs;
    }

    // ---- Flatten constraints ----
    std::vector<DevConstraint> h_constraints(n_c);
    for (int ci = 0; ci < n_c; ++ci) {
        h_constraints[ci].n_neighbors = (int)constraints[ci].neighbors.size();
        h_constraints[ci].needed      = constraints[ci].needed;
        for (int j = 0; j < h_constraints[ci].n_neighbors; ++j)
            h_constraints[ci].neighbors[j] = constraints[ci].neighbors[j];
    }

    // ---- Build per-boundary-position dependency lists ----
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

    // ---- Allocate device memory ----
    int *d_boundary, *d_interior, *d_hits, *d_valid;
    DevConstraint* d_constraints;
    DevCellDeps*   d_cell_deps;

    CUDA_CHECK(cudaMalloc(&d_boundary,    n_b * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_interior,    n_i * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_constraints, n_c * sizeof(DevConstraint)));
    CUDA_CHECK(cudaMalloc(&d_cell_deps,   n_b * sizeof(DevCellDeps)));
    CUDA_CHECK(cudaMalloc(&d_hits,        N   * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_valid,             sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_boundary,    boundary.data(),       n_b * sizeof(int),           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_interior,    interior.data(),       n_i * sizeof(int),           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_constraints, h_constraints.data(),  n_c * sizeof(DevConstraint), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cell_deps,   h_cell_deps.data(),    n_b * sizeof(DevCellDeps),   cudaMemcpyHostToDevice));
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
