#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -J MC_cuda
#SBATCH -t 60
#SBATCH --gres=gpu:1 -c 1
#SBATCH -o cuda.out -e cuda.err
#SBATCH --mem=8G

module load nvidia/cuda/13.0.0

cd /srv/home/krasselt/repo759/final759/cpp/

# Build CUDA target
cmake -S . -B build_cuda -DCMAKE_BUILD_TYPE=Release > /dev/null
cmake --build build_cuda --target minesweeper_cuda > /dev/null

echo "=== CUDA Monte Carlo Minesweeper Solver ==="
echo ""

# --- Win rate across board sizes (fixed samples) ---
echo "--- Win Rate by Board Size (500 samples/move, 100 games) ---"
echo "beginner (9x9, 10 mines):"
./build_cuda/minesweeper_cuda --rows 9  --cols 9  --mines 10 --samples 500 --games 100 --quiet

echo "intermediate (16x16, 40 mines):"
./build_cuda/minesweeper_cuda --rows 16 --cols 16 --mines 40 --samples 500 --games 100 --quiet

echo "expert (16x30, 99 mines):"
./build_cuda/minesweeper_cuda --rows 16 --cols 30 --mines 99 --samples 500 --games 100 --quiet

echo ""

# --- Throughput scaling by sample count ---
echo "--- Throughput Scaling by Sample Count (9x9, 10 mines, 50 games) ---"
for samples in 100 250 500 1000 2500 5000 10000; do
    echo "samples=$samples:"
    ./build_cuda/minesweeper_cuda --rows 9 --cols 9 --mines 10 --samples $samples --games 50 --quiet
done

echo ""

echo "--- Throughput Scaling by Sample Count (16x30, 99 mines, 20 games) ---"
for samples in 100 250 500 1000 2500 5000 10000; do
    echo "samples=$samples:"
    ./build_cuda/minesweeper_cuda --rows 16 --cols 30 --mines 99 --samples $samples --games 20 --quiet
done
