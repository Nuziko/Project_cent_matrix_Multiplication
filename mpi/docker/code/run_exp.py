import subprocess
import json
import pandas as pd

# ==========================================
# Experiment settings (edit here)
# ==========================================
KERNEL = "numba"            # "numba" (main experiment) or "blas" (reference)
MAP_BY_NODE = False         # True = spread ranks round-robin over the containers
matrix_sizes = [1024, 2048, 4096]
process_counts = [1, 2, 4, 8, 16, 32, 64]   # p = 1 must come first (it is T1)
reps = 4
TIMEOUT = 1800              # seconds allowed for one mpirun


def run_suite():
    results = []
    placement_name = "spread" if MAP_BY_NODE else "default"
    csv_filename = f"mpi_benchmark_results.csv"

    print("==================================================================")
    print("           STARTING AUTOMATED MPI BENCHMARK SUITE                 ")
    print(f"           kernel = {KERNEL} | placement = {placement_name}")
    print("==================================================================")

    for n in matrix_sizes:
        print(f"\n>>> Running for Matrix Size n = {n} <<<")
        t1_reference = None   # T1 = MPI run with p = 1, same n

        for p in process_counts:
            if n % p != 0:
                print(f"[!] Skipping p={p}: n={n} is not divisible by p")
                continue

            # Build mpirun command with oversubscription enabled
            cmd = [
                "mpirun",
                "-n", str(p),
                "--oversubscribe",
                "--allow-run-as-root",
                "--hostfile", "hostfile",
            ]
            if MAP_BY_NODE:
                cmd += ["--map-by", "node"]
            # make sure remote ranks also run with one thread
            for var in ["OMP_NUM_THREADS", "MKL_NUM_THREADS",
                        "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"]:
                cmd += ["-x", f"{var}=1"]
            cmd += [
                "python3", "worker.py",
                "--n", str(n),
                "--reps", str(reps),
                "--kernel", KERNEL
            ]

            print(f"[*] Executing: mpirun -n {p} (n={n}, {reps} repetitions)...")
            try:
                proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                      text=True, timeout=TIMEOUT)
            except subprocess.TimeoutExpired:
                print(f"[-] Timeout for p={p}, n={n}. Run 'pkill -f worker.py' in both containers.")
                continue

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

            # Speedup S = T1 / Tp and efficiency E = S / p
            if p == 1:
                t1_reference = data["median_total"]

            if t1_reference is not None:
                speedup = t1_reference / data["median_total"]
                efficiency = speedup / p
            else:
                speedup = efficiency = float("nan")

            # Total communication time = bcast + scatter + gather
            comm_time = data["bcast_median"] + data["scatter_median"] + data["gather_median"]

            data["speedup"] = speedup
            data["efficiency"] = efficiency
            data["comm_total_median"] = comm_time
            data["comm_share"] = comm_time / data["median_total"]

            print(f"    -> [p={p}] Total: {data['median_total']:.4f}s | "
                  f"Compute: {data['compute_median']:.4f}s | "
                  f"Comm: {comm_time:.4f}s | "
                  f"Speedup: {speedup:.2f}x | Efficiency: {efficiency:.2%} | "
                  f"Placement: {data['placement']}")

            results.append(data)
            pd.DataFrame(results).to_csv(csv_filename, index=False)  # keep partial results

    # ==========================================
    # Save Results to CSV
    # ==========================================
    df = pd.DataFrame(results)
    df.to_csv(csv_filename, index=False)
    print(f"\n[+] Raw results saved to: {csv_filename}")


if __name__ == "__main__":
    run_suite()
