import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CSV_FILE = "mpi_benchmark_results.csv"

CORES = 4             # physical cores of the host

markers = ['o', 's', '^']
colors = ['#1f77b4', '#d62728', '#2ca02c']


def generate_mpi_plots(df,tag):
    df = df.sort_values(["n", "p"])
    matrix_sizes = sorted(df["n"].unique())
    p_values = sorted(df["p"].unique())
    p_max = max(p_values)

    # ---------------- Figure 1: speedup and efficiency ----------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax1.plot(p_values, p_values, 'k--', alpha=0.7, label="Ideal (linear)")
    ax2.axhline(1.0, color='k', linestyle='--', alpha=0.7, label="Ideal (100%)")

    for idx, n in enumerate(matrix_sizes):
        s = df[df["n"] == n]
        ax1.plot(s["p"], s["speedup"], marker=markers[idx], color=colors[idx],
                 linewidth=2, label=f"n = {n}")
        ax2.plot(s["p"], s["efficiency"], marker=markers[idx], color=colors[idx],
                 linewidth=2, label=f"n = {n}")

    for ax in (ax1, ax2):
        ax.set_xscale("log", base=2)
        ax.set_xticks(p_values)
        ax.set_xticklabels([str(p) for p in p_values])
        ax.minorticks_off()
        ax.axvline(CORES, color="gray", linestyle=":")
        ax.axvspan(CORES, p_max * 1.1, color="gray", alpha=0.12,
                   label=f"oversubscribed (p > {CORES})")
        ax.set_xlabel("Number of MPI ranks (p)")
        ax.grid(True, linestyle="--", alpha=0.6)

    ax1.set_ylim(0, 8)
    ax1.set_title("Speedup vs. ranks", fontsize=12, fontweight='bold')
    ax1.set_ylabel(r"Speedup  $S = T_1 / T_p$")
    ax1.legend(fontsize=8)

    ax2.set_ylim(0, 1.25)
    ax2.set_title("Efficiency vs. ranks", fontsize=12, fontweight='bold')
    ax2.set_ylabel(r"Efficiency  $E = S / p$")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    f1 = f"mpi_performance_plots_{tag}.png"
    plt.savefig(f1, dpi=300)
    plt.close()
    print(f"[+] Performance plots saved as: {f1}")

    # ---------------- Figure 2: time breakdown (4 phases) ----------------
    fig, axes = plt.subplots(1, len(matrix_sizes), figsize=(13, 4.2), sharey=False)
    phases = [("bcast_median",   "Broadcast", "#1f77b4"),
              ("scatter_median", "Scatter",   "#ff7f0e"),
              ("compute_median", "Compute",   "#2ca02c"),
              ("gather_median",  "Gather",    "#d62728")]

    for i, (ax, n) in enumerate(zip(axes, matrix_sizes)):
        s = df[df["n"] == n].sort_values("p")
        x = np.arange(len(s))
        bottom = np.zeros(len(s))
        for col, label, color in phases:
            vals = s[col].values
            ax.bar(x, vals, 0.6, bottom=bottom, label=label, color=color)
            bottom += vals
        # time not counted in any phase (barriers, waiting for the slowest rank)
        other = np.clip(s["median_total"].values - bottom, 0, None)
        ax.bar(x, other, 0.6, bottom=bottom, label="Other (barriers/wait)",
               color="lightgray", hatch="//", edgecolor="gray")

        ax.set_title(f"n = {n}", fontsize=12, fontweight='bold')
        ax.set_xlabel("Number of MPI ranks (p)")
        ax.set_xticks(x)
        ax.set_xticklabels([str(p) for p in s["p"]])
        ax.grid(True, linestyle="--", alpha=0.6, axis='y')
        if i == 0:
            ax.set_ylabel("Median time (s)")
            ax.legend(fontsize=8)

    plt.tight_layout()
    f2 = f"mpi_time_breakdown_{tag}.png"
    plt.savefig(f2, dpi=300)
    plt.close()
    print(f"[+] Time breakdown plots saved as: {f2}")


if __name__ == "__main__":
    import sys 
    if len(sys.argv) < 2:
        print("Usage: python3 generate_plots.py <tag> ")
        print("Example: python3 generate_plots.py docker")
        sys.exit(1)

    tag = sys.argv[1]
    df = pd.read_csv(CSV_FILE)
    generate_mpi_plots(df,tag)

