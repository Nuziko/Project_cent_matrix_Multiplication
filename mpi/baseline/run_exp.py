import subprocess
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def run_suite():
    matrix_sizes = [1024, 2048,4096]
    process_counts = [1, 2, 4, 8]
    reps = 4
    results = []

    print("==================================================================")
    print("           STARTING AUTOMATED MPI BENCHMARK SUITE                 ")
    print("==================================================================")

    for n in matrix_sizes:
        print(f"\n>>> Running for Matrix Size n = {n} <<<")
        t1_reference = None

        for p in process_counts:
            # Build mpirun command with oversubscription enabled
            cmd = [
                "mpirun",
                "-n", str(p),
                "--oversubscribe",
                "--allow-run-as-root",
                "python", "worker.py",
                "--n", str(n),
                "--reps", str(reps)
            ]

            print(f"[*] Executing: mpirun -n {p} (n={n}, 4 repetitions)...")
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            if proc.returncode != 0:
                print(f"[-] Execution failed for p={p}, n={n}:\n{proc.stderr}")
                continue

            # Extract output JSON
            data = None
            for line in proc.stdout.splitlines():
                if line.startswith("__RESULT_JSON__"):
                    data = json.loads(line.replace("__RESULT_JSON__", ""))
                    break

            if not data:
                print(f"[-] Could not parse output for p={p}, n={n}")
                continue

            # Compute Speedup and Efficiency
            if p == 1:
                t1_reference = data["median_total"]
                speedup = 1.0
                efficiency = 1.0
            else:
                speedup = t1_reference / data["median_total"]
                efficiency = speedup / p

            # Calculate total communication time
            comm_time = data["bcast_median"] + data["scatter_median"] + data["gather_median"]

            data["speedup"] = speedup
            data["efficiency"] = efficiency
            data["comm_total_median"] = comm_time

            print(f"    -> [p={p}] Total: {data['median_total']:.4f}s | "
                  f"Compute: {data['compute_median']:.4f}s | "
                  f"Comm: {comm_time:.4f}s | "
                  f"Speedup: {speedup:.2f}x | Efficiency: {efficiency:.2%}")

            results.append(data)

    # ==========================================
    # Save Results to CSV
    # ==========================================
    df = pd.DataFrame(results)
    csv_filename = "mpi_benchmark_results.csv"
    df.to_csv(csv_filename, index=False)
    print(f"\n[+] Raw results saved to: {csv_filename}")


if __name__ == "__main__":
    run_suite()