#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -J MC_openmp
#SBATCH -c 8
#SBATCH -o openmp.out -e openmp.err
#SBATCH --mem=4G

cd /srv/home/krasselt/final759/cpp/
rm -rf build build_omp

# Build with OpenMP enabled — uncomment the three lines in CMakeLists.txt first,
# or pass the flags directly here via an override:
cmake -S . -B build_omp -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS="-fopenmp" \
    -DCMAKE_EXE_LINKER_FLAGS="-fopenmp" > /dev/null
cmake --build build_omp --target minesweeper_mc > /dev/null

echo "=== OpenMP Monte Carlo Minesweeper Solver ==="
echo "Cores available: $SLURM_CPUS_PER_TASK"
echo ""

# --- Win rate across board sizes (fixed samples) ---
echo "--- Win Rate by Board Size (500 samples/move, 100 games) ---"
echo "beginner (9x9, 10 mines):"
./build_omp/minesweeper_mc --rows 9  --cols 9  --mines 10 --samples 500 --games 100 --quiet

echo "intermediate (16x16, 40 mines):"
./build_omp/minesweeper_mc --rows 16 --cols 16 --mines 40 --samples 500 --games 100 --quiet

echo "expert (16x30, 99 mines):"
./build_omp/minesweeper_mc --rows 16 --cols 30 --mines 99 --samples 500 --games 100 --quiet

echo ""

# --- Thread scaling (strong scaling): fix work, vary OMP_NUM_THREADS ---
echo "--- Thread Scaling (9x9, 10 mines, 500 samples, 50 games) ---"
for threads in 1 2 4 8; do
    export OMP_NUM_THREADS=$threads
    echo "threads=$threads:"
    ./build_omp/minesweeper_mc --rows 9 --cols 9 --mines 10 --samples 500 --games 50 --quiet
done

echo ""

# --- Throughput scaling by sample count ---
export OMP_NUM_THREADS=8
echo "--- Throughput Scaling by Sample Count (9x9, 10 mines, 50 games, 8 threads) ---"
for samples in 100 250 500 1000 2500 5000; do
    echo "samples=$samples:"
    ./build_omp/minesweeper_mc --rows 9 --cols 9 --mines 10 --samples $samples --games 50 --quiet
done

echo ""

echo "--- Throughput Scaling by Sample Count (16x30, 99 mines, 20 games, 8 threads) ---"
for samples in 100 250 500 1000 2500 5000; do
    echo "samples=$samples:"
    ./build_omp/minesweeper_mc --rows 16 --cols 30 --mines 99 --samples $samples --games 20 --quiet
done
