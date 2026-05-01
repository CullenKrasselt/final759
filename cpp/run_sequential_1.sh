#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -J MC_seq_1
#SBATCH -c 1
#SBATCH -o sequential_1.out -e sequential_1.err
#SBATCH --mem=4G

cd /srv/home/krasselt/final759/cpp/
export OMP_NUM_THREADS=1

rm -rf build && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release > /dev/null && cmake --build build --target minesweeper_mc > /dev/null

echo "=== Sequential: Win Rate by Board Size (500 samples, 100 games) ==="

echo "beginner (9x9, 10 mines):"
./build/minesweeper_mc --rows 9  --cols 9  --mines 10 --samples 500 --games 100 --quiet

echo "intermediate (16x16, 40 mines):"
./build/minesweeper_mc --rows 16 --cols 16 --mines 40 --samples 500 --games 100 --quiet

echo "expert (16x30, 99 mines):"
./build/minesweeper_mc --rows 16 --cols 30 --mines 99 --samples 500 --games 100 --quiet
