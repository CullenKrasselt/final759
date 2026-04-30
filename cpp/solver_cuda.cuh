/**
 * solver_cuda.cuh — CUDA Monte Carlo Minesweeper solver interface.
 *
 * Drop-in parallel counterpart to MonteCarloSolver in solver.hpp.
 * The host-side preprocessing (constraint extraction, partitioning)
 * is reused from solver.cpp. Only the sample loop is offloaded to
 * the GPU: one CUDA thread per sample, iterative backtracking via
 * an explicit per-thread stack, cuRAND for randomness, shared-memory
 * reduction to accumulate mine-hit tallies.
 */
#pragma once

#include <vector>

/**
 * GPU-accelerated probability estimator.
 *
 * Inputs mirror MonteCarloSolver::get_probabilities exactly so the
 * two can be called interchangeably from main_cuda.cpp.
 *
 * Returns a vector of size rows*cols where:
 *   -1.0  => cell is not unknown
 *   [0,1] => estimated mine probability
 */
std::vector<double> cuda_get_probabilities(
    const std::vector<int>& board,
    int rows, int cols, int total_mines,
    int n_samples, int max_bt_attempts = 2000);

/**
 * Returns the flat board index of the safest unknown cell,
 * or -1 if no unknown cells remain.
 */
int cuda_best_move(
    const std::vector<int>& board,
    int rows, int cols, int total_mines,
    int n_samples, int max_bt_attempts = 2000);
