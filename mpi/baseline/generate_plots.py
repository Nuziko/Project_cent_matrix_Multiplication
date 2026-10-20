import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



def generate_mpi_plots(df):
    matrix_sizes = df["n"].unique()
    thread_counts = sorted(df["p"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(18, 5))
    markers = ['o', 's', '^', 'd']
    colors = ['#1f77b4', '#d62728', '#2ca02c']

    # 1. Speedup vs p
    ax1 = axes[0]
    ax1.plot(thread_counts, thread_counts, 'k--', label="Ideal (Linear)", alpha=0.7)
    for idx, n in enumerate(matrix_sizes):
        subset = df[df["n"] == n]
        ax1.plot(subset["p"], subset["speedup"],
                 marker=markers[idx], color=colors[idx], linewidth=2, label=f"n = {n}")
    ax1.set_title("MPI Speedup vs. Ranks (p)", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Number of Ranks (p)")
    ax1.set_ylabel("Speedup (S = T₁ / Tₚ)")
    ax1.set_xticks(thread_counts)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    # 2. Efficiency vs p
    ax2 = axes[1]
    ax2.axhline(y=1.0, color='k', linestyle='--', label="Ideal (100%)", alpha=0.7)
    for idx, n in enumerate(matrix_sizes):
        subset = df[df["n"] == n]
        ax2.plot(subset["p"], subset["efficiency"],
                 marker=markers[idx], color=colors[idx], linewidth=2, label=f"n = {n}")
    ax2.set_title("MPI Efficiency vs. Ranks (p)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Number of Ranks (p)")
    ax2.set_ylabel("Efficiency (E = S / p)")
    ax2.set_xticks(thread_counts)
    ax2.set_ylim(0, 1.25)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()
    plt.tight_layout()
    plot_file = "mpi_performance_plots.png"
    plt.savefig(plot_file, dpi=300)

    print(f"[+] Performance plots saved as: {plot_file}")

    # 3. Stacked Bar Chart: Comm vs Compute for n = 2048
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for i, ax in enumerate(axes):
        
        n_target = matrix_sizes[i]  # Select n = 1024, 2048, 4096 for each subplot
        subset_bar = df[df["n"] == n_target].sort_values("p")
        p_labels = [f"p={p}" for p in subset_bar["p"]]

        compute_times = subset_bar["compute_median"].values
        comm_times = subset_bar["comm_total_median"].values

        x = np.arange(len(p_labels))
        width = 0.45

        ax.bar(x, compute_times, width, label="Computation", color="#2ca02c")
        ax.bar(x, comm_times, width, bottom=compute_times, label="Communication (Bcast+Scatter+Gather)", color="#d62728")

        ax.set_title(f"Time Breakdown (n = {n_target})", fontsize=12, fontweight='bold')
        ax.set_xlabel("Number of Ranks (p)")
        ax.set_ylabel("Time (seconds)")
        ax.set_xticks(x)
        ax.set_xticklabels(p_labels)
        ax.grid(True, linestyle="--", alpha=0.6, axis='y')
        ax.legend()

    plt.tight_layout()
    plot_file = "mpi_time_breakdown_plots.png"
    plt.savefig(plot_file, dpi=300)
    print(f"[+] Time breakdown plots saved as: {plot_file}")
    

if __name__ == "__main__":
    df = pd.read_csv("mpi_benchmark_results.csv")
    generate_mpi_plots(df)
    