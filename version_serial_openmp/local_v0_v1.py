

import os
os.environ["NUMBA_NUM_THREADS"] = "8"
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["NUMBA_THREADING_LAYER"] = "workqueue" 
import time
import numpy as np
import pandas as pd
import numba
from numba import njit, prange, set_num_threads


@njit
def matmul_serial(A, B, C):
    n, m = A.shape
    p_dim = B.shape[1]
    for i in range(n):
        for k in range(m):
            a = A[i, k]
            for j in range(p_dim):
                C[i, j] += a * B[k, j]


@njit(parallel=True)
def matmul_parallel(A, B, C):
    n, m = A.shape
    p_dim = B.shape[1]
    for i in prange(n):
        for k in range(m):
            a = A[i, k]
            for j in range(p_dim):
                C[i, j] += a * B[k, j]


def warmup():
    print("[*] Warming up JIT compiler...")
    dummy_A = np.random.rand(4, 4).astype(np.float64)
    dummy_B = np.random.rand(4, 4).astype(np.float64)
    dummy_C = np.zeros((4, 4), dtype=np.float64)

    matmul_serial(dummy_A, dummy_B, dummy_C)
    dummy_C.fill(0.0)
    matmul_parallel(dummy_A, dummy_B, dummy_C)
    print("[+] JIT warm-up complete.\n")


def run_benchmarks():
    warmup()

    
    allocated_pool = numba.config.NUMBA_NUM_THREADS
    print(f"[*] Forced Thread Pool Size: {allocated_pool}")

    matrix_sizes = [1024,2048,4096]
    thread_counts = [1, 2, 4, 8]  
    reps = 4
    results = []

    for n in matrix_sizes:
        print(f"==================================================")
        print(f"           BENCHMARKING MATRIX SIZE n = {n}       ")
        print(f"==================================================")

        A = np.random.rand(n, n).astype(np.float64)
        B = np.random.rand(n, n).astype(np.float64)
        C = np.zeros((n, n), dtype=np.float64)

        
        C_ref = A @ B


        print(f"\n--- Running Version 0 (Serial Baseline) ---")
        v0_times = []
        for r in range(reps):
            C.fill(0.0)
            t_start = time.perf_counter()
            matmul_serial(A, B, C)
            t_end = time.perf_counter()
            assert np.allclose(C, C_ref), "[-] Error: Version 0 output mismatch!"
            v0_times.append(t_end - t_start)
            print(f"  Rep {r+1}/{reps}: {v0_times[-1]:.4f} s")

        t_serial_median = float(np.median(v0_times))
        print(f"  -> Version 0 Median Time: {t_serial_median:.4f} s")

        results.append({
            "version": "v0_serial",
            "n": n,
            "p": 1,
            "rep_1": v0_times[0],
            "rep_2": v0_times[1],
            "rep_3": v0_times[2],
            "rep_4": v0_times[3],
            "median_time": t_serial_median,
            "speedup": 1.0,
            "efficiency": 1.0
        })

      
        print(f"\n--- Running Version 1 (Numba Parallel) ---")
        for p in thread_counts:
            set_num_threads(p)
            v1_times = []

            for r in range(reps):
                C.fill(0.0)
                t_start = time.perf_counter()
                matmul_parallel(A, B, C)
                t_end = time.perf_counter()
                assert np.allclose(C, C_ref), f"[-] Error: Version 1 (p={p}) output mismatch!"
                v1_times.append(t_end - t_start)
                print(f"  [p={p}] Rep {r+1}/{reps}: {v1_times[-1]:.4f} s")

            median_time = float(np.median(v1_times))

            
            speedup = t_serial_median / median_time
           
            efficiency = speedup / p

            print(f"  -> [p={p}] Median: {median_time:.4f} s | Speedup: {speedup:.2f}x | Efficiency: {efficiency:.2%}")

            results.append({
                "version": "v1_parallel",
                "n": n,
                "p": p,
                "rep_1": v1_times[0],
                "rep_2": v1_times[1],
                "rep_3": v1_times[2],
                "rep_4": v1_times[3],
                "median_time": median_time,
                "speedup": speedup,
                "efficiency": efficiency
            })

    
    df = pd.DataFrame(results)
    output_filename = "matmul_v0_v1_results.csv"
    df.to_csv(output_filename, index=False)
    print(f"\n[+] Results successfully saved to: {output_filename}")
    print("\nSummary Table:")
    print(df[["version", "n", "p", "median_time", "speedup", "efficiency"]].to_string(index=False))

if __name__ == "__main__":
    run_benchmarks()