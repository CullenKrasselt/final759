"""
plot_results.py — Generate all result plots for the Monte Carlo Minesweeper report.
Run from the repo root:  python plot_results.py
Outputs PNG files to ./plots/
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

os.makedirs("plots", exist_ok=True)

# ── Colour palette ─────────────────────────────────────────────────────────────
C_SEQ   = "#2166ac"   # blue
C_OMP   = "#d6604d"   # red-orange
C_CUDA  = "#4dac26"   # green

# ══════════════════════════════════════════════════════════════════════════════
# 1.  Win Rate by Board Size
# ══════════════════════════════════════════════════════════════════════════════
boards   = ["Beginner\n9×9, 10 mines", "Intermediate\n16×16, 40 mines", "Expert\n16×30, 99 mines"]
seq_wr   = [94, 64,  1]
omp_wr   = [95, 76,  2]
cuda_wr  = [93, 62,  0]

x  = np.arange(len(boards))
bw = 0.25

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - bw, seq_wr,  bw, label="Sequential",    color=C_SEQ,  zorder=3)
ax.bar(x,      omp_wr,  bw, label="OpenMP (8 threads)", color=C_OMP,  zorder=3)
ax.bar(x + bw, cuda_wr, bw, label="CUDA",           color=C_CUDA, zorder=3)

for i, (s, o, c) in enumerate(zip(seq_wr, omp_wr, cuda_wr)):
    ax.text(i - bw, s + 1, f"{s}%", ha="center", va="bottom", fontsize=9)
    ax.text(i,      o + 1, f"{o}%", ha="center", va="bottom", fontsize=9)
    ax.text(i + bw, c + 1, f"{c}%", ha="center", va="bottom", fontsize=9)

ax.set_xticks(x); ax.set_xticklabels(boards)
ax.set_ylabel("Win Rate (%)")
ax.set_title("Win Rate by Board Size — 500 samples/move, 100 games")
ax.set_ylim(0, 110)
ax.yaxis.set_major_formatter(ticker.PercentFormatter())
ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
ax.legend()
fig.tight_layout()
fig.savefig("plots/win_rate_by_board.png", dpi=150)
plt.close(fig)
print("Saved plots/win_rate_by_board.png")

# ══════════════════════════════════════════════════════════════════════════════
# 2.  Time per Game (500 samples/move)
# ══════════════════════════════════════════════════════════════════════════════
seq_t  = [0.09, 3.46, 2.89]
omp_t  = [0.03, 0.81, 0.63]
cuda_t = [0.05, 1.42, 0.53]

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - bw, seq_t,  bw, label="Sequential",        color=C_SEQ,  zorder=3)
ax.bar(x,      omp_t,  bw, label="OpenMP (8 threads)", color=C_OMP,  zorder=3)
ax.bar(x + bw, cuda_t, bw, label="CUDA",               color=C_CUDA, zorder=3)

for i, (s, o, c) in enumerate(zip(seq_t, omp_t, cuda_t)):
    ax.text(i - bw, s + 0.03, f"{s:.2f}s", ha="center", va="bottom", fontsize=8)
    ax.text(i,      o + 0.03, f"{o:.2f}s", ha="center", va="bottom", fontsize=8)
    ax.text(i + bw, c + 0.03, f"{c:.2f}s", ha="center", va="bottom", fontsize=8)

ax.set_xticks(x); ax.set_xticklabels(boards)
ax.set_ylabel("Time per Game (s)")
ax.set_title("Time per Game by Board Size — 500 samples/move, 100 games")
ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
ax.legend()
fig.tight_layout()
fig.savefig("plots/time_per_game.png", dpi=150)
plt.close(fig)
print("Saved plots/time_per_game.png")

# ══════════════════════════════════════════════════════════════════════════════
# 3.  Speedup vs Sequential (500 samples/move)
# ══════════════════════════════════════════════════════════════════════════════
omp_speedup  = [s/o for s, o in zip(seq_t, omp_t)]
cuda_speedup = [s/c for s, c in zip(seq_t, cuda_t)]

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - bw/2, omp_speedup,  bw, label="OpenMP (8 threads)", color=C_OMP,  zorder=3)
ax.bar(x + bw/2, cuda_speedup, bw, label="CUDA",               color=C_CUDA, zorder=3)
ax.axhline(1.0, color="black", linewidth=1, linestyle="--", label="Sequential baseline")

for i, (o, c) in enumerate(zip(omp_speedup, cuda_speedup)):
    ax.text(i - bw/2, o + 0.1, f"{o:.1f}×", ha="center", va="bottom", fontsize=9)
    ax.text(i + bw/2, c + 0.1, f"{c:.1f}×", ha="center", va="bottom", fontsize=9)

ax.set_xticks(x); ax.set_xticklabels(boards)
ax.set_ylabel("Speedup over Sequential")
ax.set_title("Parallel Speedup vs Sequential — 500 samples/move")
ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
ax.legend()
fig.tight_layout()
fig.savefig("plots/speedup.png", dpi=150)
plt.close(fig)
print("Saved plots/speedup.png")

# ══════════════════════════════════════════════════════════════════════════════
# 4.  Throughput Scaling — Beginner Board (9×9)
# ══════════════════════════════════════════════════════════════════════════════
samples_beg = [100, 250, 500, 1000, 2500, 5000]
seq_beg     = [0.02, 0.05, 0.11, 0.23, 0.44, 0.89]
omp_beg     = [0.12, 0.01, 0.05, 0.09, 0.14, 0.31]
cuda_beg    = [0.03, 0.05, 0.05, 0.05, 0.05, 0.05]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(samples_beg, seq_beg,  "o-", color=C_SEQ,  label="Sequential",        linewidth=2, markersize=6)
ax.plot(samples_beg, omp_beg,  "s-", color=C_OMP,  label="OpenMP (8 threads)",linewidth=2, markersize=6)
ax.plot(samples_beg, cuda_beg, "^-", color=C_CUDA, label="CUDA",              linewidth=2, markersize=6)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Samples per Move")
ax.set_ylabel("Time per Game (s)")
ax.set_title("Throughput Scaling — Beginner Board (9×9, 10 mines, 50 games)")
ax.grid(True, which="both", linestyle="--", alpha=0.4)
ax.legend()
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
fig.tight_layout()
fig.savefig("plots/throughput_beginner.png", dpi=150)
plt.close(fig)
print("Saved plots/throughput_beginner.png")

# ══════════════════════════════════════════════════════════════════════════════
# 5.  Throughput Scaling — Expert Board (16×30)
# ══════════════════════════════════════════════════════════════════════════════
samples_exp_seq  = [100, 250, 500, 1000]
seq_exp          = [0.14, 0.70, 4.51, 13.70]

samples_exp_omp  = [100, 250, 500, 1000, 2500, 5000]
omp_exp          = [0.20, 0.37, 0.50, 2.28, 6.27, 31.98]

samples_exp_cuda = [100, 250, 500, 1000, 2500, 5000, 10000]
cuda_exp         = [0.10, 0.33, 0.57, 0.88, 1.80, 3.46, 3.83]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(samples_exp_seq,  seq_exp,  "o-", color=C_SEQ,  label="Sequential",        linewidth=2, markersize=6)
ax.plot(samples_exp_omp,  omp_exp,  "s-", color=C_OMP,  label="OpenMP (8 threads)",linewidth=2, markersize=6)
ax.plot(samples_exp_cuda, cuda_exp, "^-", color=C_CUDA, label="CUDA",              linewidth=2, markersize=6)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Samples per Move")
ax.set_ylabel("Time per Game (s)")
ax.set_title("Throughput Scaling — Expert Board (16×30, 99 mines, 20 games)")
ax.grid(True, which="both", linestyle="--", alpha=0.4)
ax.legend()
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
fig.tight_layout()
fig.savefig("plots/throughput_expert.png", dpi=150)
plt.close(fig)
print("Saved plots/throughput_expert.png")

# ══════════════════════════════════════════════════════════════════════════════
# 6.  OpenMP Thread Scaling
# ══════════════════════════════════════════════════════════════════════════════
threads     = [1, 2, 4, 8]
omp_times   = [0.09, 0.04, 0.04, 0.10]
omp_speedup_t = [omp_times[0] / t for t in omp_times]
ideal_speedup = [float(t) for t in threads]

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(threads, omp_speedup_t, "o-", color=C_OMP, label="Measured speedup", linewidth=2, markersize=8)
ax.plot(threads, ideal_speedup, "k--", label="Ideal linear speedup", linewidth=1.5)

for t, s in zip(threads, omp_speedup_t):
    ax.annotate(f"{s:.1f}×", (t, s), textcoords="offset points", xytext=(5, 5), fontsize=9)

ax.set_xticks(threads)
ax.set_xlabel("Number of OpenMP Threads")
ax.set_ylabel("Speedup vs 1 Thread")
ax.set_title("OpenMP Thread Scaling — Beginner Board (9×9, 500 samples, 50 games)")
ax.grid(linestyle="--", alpha=0.5)
ax.legend()
fig.tight_layout()
fig.savefig("plots/omp_thread_scaling.png", dpi=150)
plt.close(fig)
print("Saved plots/omp_thread_scaling.png")

# ══════════════════════════════════════════════════════════════════════════════
# 7.  Win Rate vs Sample Count (Beginner Board, sequential)
# ══════════════════════════════════════════════════════════════════════════════
samples_wr  = [100, 250, 500, 1000, 2500, 5000]
seq_wr_samp = [100, 92, 94, 94, 94, 94]

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(samples_wr, seq_wr_samp, "o-", color=C_SEQ, linewidth=2, markersize=7)
ax.axhline(94, color="gray", linestyle="--", linewidth=1, label="Apparent ceiling ~94%")
for s, w in zip(samples_wr, seq_wr_samp):
    ax.text(s, w + 0.8, f"{w}%", ha="center", fontsize=9)
ax.set_xscale("log")
ax.set_xlabel("Samples per Move")
ax.set_ylabel("Win Rate (%)")
ax.set_ylim(85, 105)
ax.yaxis.set_major_formatter(ticker.PercentFormatter())
ax.set_title("Win Rate vs Sample Count — Beginner Board (9×9, 10 mines, 50 games, Sequential)")
ax.grid(True, which="both", linestyle="--", alpha=0.4)
ax.legend()
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
fig.tight_layout()
fig.savefig("plots/win_rate_vs_samples.png", dpi=150)
plt.close(fig)
print("Saved plots/win_rate_vs_samples.png")

print("\nAll plots saved to ./plots/")
