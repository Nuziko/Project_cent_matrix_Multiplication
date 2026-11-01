import os

# Must be set BEFORE importing numpy / numba (SSH sessions do not inherit
# the Docker ENV, so we set them here for every rank).
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"

import sys
import json
import socket
import argparse
from collections import Counter

import numpy as np
from numba import njit
from mpi4py import MPI


# ----------------------------------------------------------------------
# Local kernels (same i-k-j loop as Versions 0 and 1, single thread)
# ----------------------------------------------------------------------
@njit
def matmul_numba(A, B, C):
    n, m = A.shape
    p_dim = B.shape[1]
    for i in range(n):
        for k in range(m):
            a = A[i, k]
            for j in range(p_dim):
                C[i, j] += a * B[k, j]


def matmul_blas(A, B, C):
    np.matmul(A, B, out=C)


def one_repetition(comm, kernel, A, B, C, A_loc, C_loc):
    """Run one full distributed multiplication.
    Returns this rank's times [bcast, scatter, compute, gather, total]."""
    C_loc.fill(0.0)                       # kernel accumulates with +=

    comm.Barrier()
    t_start = MPI.Wtime()

    t0 = MPI.Wtime()
    comm.Bcast(B, root=0)
    t_bcast = MPI.Wtime() - t0

    comm.Barrier()
    t0 = MPI.Wtime()
    comm.Scatter(A, A_loc, root=0)
    t_scatter = MPI.Wtime() - t0

    comm.Barrier()
    t0 = MPI.Wtime()
    kernel(A_loc, B, C_loc)
    t_compute = MPI.Wtime() - t0

    comm.Barrier()
    t0 = MPI.Wtime()
    comm.Gather(C_loc, C, root=0)
    t_gather = MPI.Wtime() - t0

    comm.Barrier()
    t_total = MPI.Wtime() - t_start

    return np.array([t_bcast, t_scatter, t_compute, t_gather, t_total])


def run_worker():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1024, help="Matrix dimension n x n")
    parser.add_argument("--reps", type=int, default=4, help="Timed repetitions")
    parser.add_argument("--kernel", choices=["numba", "blas"], default="numba",
                        help="Local compute kernel")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    n, reps = args.n, args.reps
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # All ranks take the same decision, so nobody is left waiting.
    if n % size != 0:
        if rank == 0:
            sys.stderr.write(f"[-] n={n} must be divisible by p={size}\n")
        sys.exit(1)

    rows = n // size
    kernel = matmul_numba if args.kernel == "numba" else matmul_blas

    # --- placement: how many ranks run on each host ---
    hosts = comm.gather(socket.gethostname(), root=0)

    # --- JIT warm-up on every rank (4x4), untimed ---
    d = np.random.rand(4, 4)
    kernel(d, d, np.zeros((4, 4)))
    comm.Barrier()

    # --- data (generated once, not timed) ---
    if rank == 0:
        rng = np.random.default_rng(args.seed)
        A = rng.random((n, n))
        B = rng.random((n, n))
        C = np.empty((n, n))
    else:
        A, C = None, None
        B = np.empty((n, n))
    A_loc = np.empty((rows, n))
    C_loc = np.empty((rows, n))

    # --- one full untimed repetition: warms MPI connections, page tables,
    #     caches, and is used for the correctness check ---
    one_repetition(comm, kernel, A, B, C, A_loc, C_loc)
    ok = True
    if rank == 0:
        ok = bool(np.allclose(C, A @ B))
    ok = comm.bcast(ok, root=0)
    if not ok:
        if rank == 0:
            sys.stderr.write("[-] Output matrix C does not match A @ B\n")
        sys.exit(2)

    # --- timed repetitions ---
    all_reps = []
    for _ in range(reps):
        local = one_repetition(comm, kernel, A, B, C, A_loc, C_loc)
        maxed = np.zeros(5) if rank == 0 else None
        comm.Reduce(local, maxed, op=MPI.MAX, root=0)   # slowest rank
        if rank == 0:
            all_reps.append(maxed)

    if rank == 0:
        R = np.array(all_reps)                  # shape (reps, 5)
        med = np.median(R, axis=0)
        payload = {
            "kernel": args.kernel,
            "n": n,
            "p": size,
            "placement": ";".join(f"{h}:{c}" for h, c in sorted(Counter(hosts).items())),
        }
        for i in range(reps):
            payload[f"rep_{i + 1}"] = float(R[i, 4])
        payload.update({
            "median_total": float(med[4]),
            "bcast_median": float(med[0]),
            "scatter_median": float(med[1]),
            "compute_median": float(med[2]),
            "gather_median": float(med[3]),
        })
        print(f"__RESULT_JSON__{json.dumps(payload)}")


if __name__ == "__main__":
    run_worker()
