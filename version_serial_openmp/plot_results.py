import matplotlib.pyplot as plt
import pandas as pd

def generate_plots(csv_path="matmul_v0_v1_results.csv"):
    df = pd.read_csv(csv_path)

    v1_df = df[df["version"] == "v1_parallel"]
    matrix_sizes = v1_df["n"].unique()
    thread_counts = [1, 2, 4, 8]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    markers = ['o', 's', '^', 'd']
    colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e']

    
    ax1 = axes[0]
    ax1.plot(thread_counts, thread_counts, 'k--', label="Ideal (Linear)", alpha=0.7)
    for idx, n in enumerate(matrix_sizes):
        subset = v1_df[v1_df["n"] == n]
        ax1.plot(subset["p"], subset["speedup"],
                 marker=markers[idx % len(markers)],
                 color=colors[idx % len(colors)],
                 linewidth=2, label=f"n = {n}")

    ax1.set_title("Speedup vs. Threads (p)", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Number of Threads (p)")
    ax1.set_ylabel("Speedup (S = T₁ / Tₚ)")
    ax1.set_xticks(thread_counts)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    
    ax2 = axes[1]
    ax2.axhline(y=1.0, color='k', linestyle='--', label="Ideal (100%)", alpha=0.7)
    for idx, n in enumerate(matrix_sizes):
        subset = v1_df[v1_df["n"] == n]
        ax2.plot(subset["p"], subset["efficiency"],
                 marker=markers[idx % len(markers)],
                 color=colors[idx % len(colors)],
                 linewidth=2, label=f"n = {n}")

    ax2.set_title("Efficiency vs. Threads (p)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Number of Threads (p)")
    ax2.set_ylabel("Efficiency (E = S / p)")
    ax2.set_xticks(thread_counts)
    ax2.set_ylim(0, 1.25)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

    
    ax3 = axes[2]
    for idx, n in enumerate(matrix_sizes):
        subset = v1_df[v1_df["n"] == n]
        ax3.plot(subset["p"], subset["median_time"],
                 marker=markers[idx % len(markers)],
                 color=colors[idx % len(colors)],
                 linewidth=2, label=f"n = {n}")

    ax3.set_title("Execution Time vs. Threads (p)", fontsize=12, fontweight='bold')
    ax3.set_xlabel("Number of Threads (p)")
    ax3.set_ylabel("Time (seconds)")
    ax3.set_xticks(thread_counts)
    ax3.grid(True, linestyle="--", alpha=0.6)
    ax3.legend()

    plt.tight_layout()
    plot_file = "v0_v1_oversubscribed_plots.png"
    plt.savefig(plot_file, dpi=300)
    print(f"[+] Plots successfully generated and saved as '{plot_file}'")
    plt.show()

if __name__ == "__main__":
    generate_plots()