#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -J MC_seq_2
#SBATCH -c 1
#SBATCH -o sequential_2.out -e sequential_2.err
#SBATCH --mem=4G

cd /srv/home/krasselt/final759/cpp/
export OMP_NUM_THREADS=1

rm -rf build && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release > /dev/null && cmake --build build --target minesweeper_mc > /dev/null

echo "=== Sequential: Throughput Scaling (9x9, 10 mines, 50 games) ==="
for samples in 100 250 500 1000 2500 5000; do
    echo "samples=$samples:"
    ./build/minesweeper_mc --rows 9 --cols 9 --mines 10 --samples $samples --games 50 --quiet
done

echo ""
echo "=== Sequential: Throughput Scaling (16x30, 99 mines, 20 games) ==="
for samples in 100 250 500 1000; do
    echo "samples=$samples:"
    ./build/minesweeper_mc --rows 16 --cols 30 --mines 99 --samples $samples --games 20 --quiet
done
