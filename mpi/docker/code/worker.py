import os
# Prevent BLAS from creating background threads inside each MPI rank
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import json
import argparse
import numpy as np
from mpi4py import MPI

def run_worker():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1024, help="Matrix dimension n x n")
    parser.add_argument("--reps", type=int, default=4, help="Number of repetitions")
    args = parser.parse_args()

    n = args.n
    reps = args.reps

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if n % size != 0:
        if rank == 0:
            sys.stderr.write(f"[-] Matrix dimension {n} must be divisible by process count {size}\n")
        sys.exit(1)

    rows_per_proc = n // size

    # ==========================================
    # Warm-up (Compiles/allocates MPI buffers)
    # ==========================================
    dummy_b = np.empty((2, 2), dtype=np.float64)
    dummy_a = np.empty((2 // size, 2), dtype=np.float64) if (2 % size == 0) else None
    comm.Bcast(dummy_b, root=0)

    # Buffers to record 4 repetitions
    rep_total = []
    rep_bcast = []
    rep_scatter = []
    rep_compute = []
    rep_gather = []

    for r in range(reps):
        if rank == 0:
            A = np.random.rand(n, n).astype(np.float64)
            B = np.random.rand(n, n).astype(np.float64)
            C = np.empty((n, n), dtype=np.float64)
        else:
            A = None
            B = np.empty((n, n), dtype=np.float64)
            C = None

        A_loc = np.empty((rows_per_proc, n), dtype=np.float64)
        C_loc = np.empty((rows_per_proc, n), dtype=np.float64)

        # ----------------------------------------------------
        # 1. Total Timer Start
        # ----------------------------------------------------
        comm.Barrier()
        t_total_start = MPI.Wtime()

        # ----------------------------------------------------
        # 2. Broadcast Matrix B
        # ----------------------------------------------------
        comm.Barrier()
        t0 = MPI.Wtime()
        comm.Bcast(B, root=0)
        t_bcast = MPI.Wtime() - t0

        # ----------------------------------------------------
        # 3. Scatter Matrix A
        # ----------------------------------------------------
        comm.Barrier()
        t0 = MPI.Wtime()
        comm.Scatter(A, A_loc, root=0)
        t_scatter = MPI.Wtime() - t0

        # ----------------------------------------------------
        # 4. Computation (Local Kernel)
        # ----------------------------------------------------
        comm.Barrier()
        t0 = MPI.Wtime()
        C_loc = A_loc @ B
        t_compute = MPI.Wtime() - t0

        # ----------------------------------------------------
        # 5. Gather C_loc back to C
        # ----------------------------------------------------
        comm.Barrier()
        t0 = MPI.Wtime()
        comm.Gather(C_loc, C, root=0)
        t_gather = MPI.Wtime() - t0

        comm.Barrier()
        t_total = MPI.Wtime() - t_total_start

        # Take max time across all ranks (true parallel accounting)
        max_bcast = comm.reduce(t_bcast, op=MPI.MAX, root=0)
        max_scatter = comm.reduce(t_scatter, op=MPI.MAX, root=0)
        max_compute = comm.reduce(t_compute, op=MPI.MAX, root=0)
        max_gather = comm.reduce(t_gather, op=MPI.MAX, root=0)
        max_total = comm.reduce(t_total, op=MPI.MAX, root=0)

        if rank == 0:
            # Correctness verification on repetition 1
            if r == 0:
                C_ref = A @ B
                assert np.allclose(C, C_ref), "[-] Error: Output matrix C mismatch!"

            rep_total.append(max_total)
            rep_bcast.append(max_bcast)
            rep_scatter.append(max_scatter)
            rep_compute.append(max_compute)
            rep_gather.append(max_gather)

    # Rank 0 returns the timing summary
    if rank == 0:
        result_payload = {
            "n": n,
            "p": size,
            "rep_1": rep_total[0],
            "rep_2": rep_total[1],
            "rep_3": rep_total[2],
            "rep_4": rep_total[3],
            "median_total": float(np.median(rep_total)),
            "bcast_median": float(np.median(rep_bcast)),
            "scatter_median": float(np.median(rep_scatter)),
            "compute_median": float(np.median(rep_compute)),
            "gather_median": float(np.median(rep_gather)),
        }
        # Print output prefixed with an identifier for easy JSON parsing
        print(f"__RESULT_JSON__{json.dumps(result_payload)}")

if __name__ == "__main__":
    run_worker()