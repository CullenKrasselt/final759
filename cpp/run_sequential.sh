#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -J MC_sequential
#SBATCH -t 60
#SBATCH -c 1
#SBATCH -o sequential.out -e sequential.err
#SBATCH --mem=4G

cd /srv/home/krasselt/repo759/final759/cpp/

# Build sequential target
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release > /dev/null
cmake --build build --target minesweeper_mc > /dev/null

echo "=== Sequential C++ Monte Carlo Minesweeper Solver ==="
echo ""

# --- Win rate across board sizes (fixed samples) ---
echo "--- Win Rate by Board Size (500 samples/move, 100 games) ---"
echo "beginner (9x9, 10 mines):"
./build/minesweeper_mc --rows 9  --cols 9  --mines 10 --samples 500 --games 100 --quiet

echo "intermediate (16x16, 40 mines):"
./build/minesweeper_mc --rows 16 --cols 16 --mines 40 --samples 500 --games 100 --quiet

echo "expert (16x30, 99 mines):"
./build/minesweeper_mc --rows 16 --cols 30 --mines 99 --samples 500 --games 100 --quiet

echo ""

# --- Scaling: time vs sample count (beginner board, 50 games) ---
echo "--- Throughput Scaling by Sample Count (9x9, 10 mines, 50 games) ---"
for samples in 100 250 500 1000 2500 5000; do
    echo "samples=$samples:"
    ./build/minesweeper_mc --rows 9 --cols 9 --mines 10 --samples $samples --games 50 --quiet
done

echo ""

# --- Scaling: time vs sample count (expert board, 20 games) ---
echo "--- Throughput Scaling by Sample Count (16x30, 99 mines, 20 games) ---"
for samples in 100 250 500 1000 2500 5000; do
    echo "samples=$samples:"
    ./build/minesweeper_mc --rows 16 --cols 30 --mines 99 --samples $samples --games 20 --quiet
done
