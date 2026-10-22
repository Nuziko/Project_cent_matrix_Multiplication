# MPI matrix multiplication baseline

This directory contains a baseline MPI implementation of dense square matrix
multiplication, its benchmark output, and plots summarizing the measurements.
The experiment was run on a CPU with **4 cores**, as reported for this run.
The benchmark uses MPI **processes (ranks)**, not threads, and tests rank counts
from 1 through 64. Consequently, runs with more than four ranks oversubscribe
the available CPU cores. The experiment is useful for showing the effects of
communication and oversubscription; it should not be read as a scaling test on
a 64-core machine.

## Files

- [`worker.py`](./worker.py) runs the MPI matrix multiplication and measures
  its phases.
- [`run_exp.py`](./run_exp.py) launches the benchmark configurations and
  writes the CSV and log output.
- [`mpi_benchmark_results.csv`](./mpi_benchmark_results.csv) contains the four
  individual total-time samples, median timings, speedups, and efficiencies.
- [`logs.txt`](./logs.txt) is the human-readable summary emitted by the run.
- [`generate_plots.py`](./generate_plots.py) generates the figures below.
- [`mpi_performance_plots.png`](./mpi_performance_plots.png) plots speedup and
  efficiency against MPI rank count.
- [`mpi_time_breakdown_plots.png`](./mpi_time_breakdown_plots.png) compares
  median computation and communication times.

## What was measured

The program multiplies two `n × n` `float64` matrices for `n = 1024`, `2048`,
and `4096`, using `p = 1`, `2`, `4`, `8`, `16`, `32`, and `64` MPI ranks.
Each configuration is repeated four times, and the reported timings are
medians. The worker limits OpenMP, MKL, and OpenBLAS to one thread per MPI rank,
so these results are not intended to combine MPI ranks with multithreaded BLAS.

For each repetition, rank 0 creates the input matrices. The implementation:

1. Broadcasts the full matrix `B` to all ranks.
2. Scatters row blocks of `A` among the ranks.
3. Computes each local block with `A_loc @ B`.
4. Gathers the resulting row blocks into `C` on rank 0.

The worker checks the first gathered result against a serial `A @ B` result.
The timings recorded for each repetition are the maximum measured time across
ranks, which reflects the slowest rank rather than just rank 0's local work.
The total includes the timed MPI operations and computation, with barriers
used to synchronize phase boundaries.

The CSV and log report:

- **Total time:** median elapsed time of the complete operation, in seconds.
- **Computation:** median time for the local matrix multiplication.
- **Communication:** the sum of median broadcast, scatter, and gather times.
- **Speedup:** `T₁ / Tₚ`, where `T₁` is the one-rank median total time for the
  same matrix size and `Tₚ` is the median total time with `p` ranks.
- **Efficiency:** `speedup / p`. Multiply by 100 to express it as a percentage.

The one-rank case is the MPI baseline for each matrix size, so its speedup and
efficiency are 1.0 (100%) by definition. They are not comparisons with a
separate non-MPI implementation.

## Results

All times below are medians in seconds. Communication is the sum of the
broadcast, scatter, and gather median timings in the CSV. Speedup and
efficiency are calculated from **total** time, not compute time alone.

### Matrix size `n = 1024`

| Ranks (`p`) | Total (s) | Compute (s) | Communication (s) | Speedup | Efficiency |
|---:|---:|---:|---:|---:|---:|
| 1  | 0.0542 | 0.0529 | 0.0013 | 1.00× | 100.00% |
| 2  | 0.0349 | 0.0291 | 0.0064 | 1.55× | 77.65% |
| 4  | 0.0587 | 0.0332 | 0.0177 | 0.92× | 23.08% |
| 8  | 0.0563 | 0.0325 | 0.0178 | 0.96× | 12.04% |
| 16 | 0.0866 | 0.0364 | 0.0331 | 0.63× | 3.91% |
| 32 | 0.0968 | 0.0305 | 0.0399 | 0.56× | 1.75% |
| 64 | 0.1464 | 0.0358 | 0.0774 | 0.37× | 0.58% |

Two ranks are fastest: total time falls from 0.0542 s to 0.0349 s, a reduction
of about 35.6%. From four ranks onward, the total is no better than the
one-rank result. At this small problem size, each rank has little computation
to do, while collective communication and MPI/process overhead remain
significant. At 64 ranks, communication alone is about 0.0774 s, over twice
the computation time, and the total is about 2.7 times the one-rank time.

### Matrix size `n = 2048`

| Ranks (`p`) | Total (s) | Compute (s) | Communication (s) | Speedup | Efficiency |
|---:|---:|---:|---:|---:|---:|
| 1  | 0.4170 | 0.4068 | 0.0106 | 1.00× | 100.00% |
| 2  | 0.2172 | 0.2002 | 0.0172 | 1.92× | 95.98% |
| 4  | 0.2486 | 0.2088 | 0.0364 | 1.68× | 41.95% |
| 8  | 0.2885 | 0.2270 | 0.0610 | 1.45× | 18.07% |
| 16 | 0.3165 | 0.2156 | 0.0918 | 1.32× | 8.24% |
| 32 | 0.4164 | 0.2621 | 0.1439 | 1.00× | 3.13% |
| 64 | 0.7221 | 0.3298 | 0.3527 | 0.58× | 0.90% |

Two ranks again give the best result: 0.2172 s, about 47.9% less than the
one-rank time. Four ranks still improve on one rank, but are slower than two.
At 32 ranks the speedup has effectively disappeared; at 64 ranks, total time
is about 1.7 times the one-rank time. Communication rises steadily with rank
count and is slightly greater than computation at 64 ranks.

### Matrix size `n = 4096`

| Ranks (`p`) | Total (s) | Compute (s) | Communication (s) | Speedup | Efficiency |
|---:|---:|---:|---:|---:|---:|
| 1  | 3.3109 | 3.2741 | 0.0364 | 1.00× | 100.00% |
| 2  | 1.7237 | 1.6624 | 0.0615 | 1.92× | 96.04% |
| 4  | 1.7854 | 1.6624 | 0.1194 | 1.85× | 46.36% |
| 8  | 1.9025 | 1.6958 | 0.2095 | 1.74× | 21.75% |
| 16 | 2.1640 | 1.7569 | 0.3522 | 1.53× | 9.56% |
| 32 | 2.4996 | 1.7875 | 0.6967 | 1.32× | 4.14% |
| 64 | 3.1880 | 1.8686 | 1.3395 | 1.04× | 1.62% |

The largest matrix shows the clearest useful parallel speedup, but two ranks
are still fastest: 1.7237 s, about 47.9% less time than the one-rank baseline.
Four ranks are close (1.7854 s), but not faster than two. The compute time is
nearly flat from two through four ranks (about 1.662 s), while communication
increases from 0.0615 s at two ranks to 0.1194 s at four. At 64 ranks,
communication is about 1.3395 s and total time has returned to almost the
one-rank time.

## Reading the plots

### Speedup and efficiency

In [`mpi_performance_plots.png`](./mpi_performance_plots.png), the dashed line
is ideal linear speedup (`S = p`), and the horizontal dashed line in the
efficiency chart is ideal 100% efficiency. The measured curves stay well below
the speedup line as rank count grows. Efficiency also drops quickly once the
run uses more ranks than the workload and hardware can use effectively.

For all three matrix sizes, the best measured speedup is at two ranks: roughly
1.55× for `n = 1024` and 1.92× for each of `n = 2048` and `n = 4096`. The
efficiency at two ranks is 77.65%, 95.98%, and 96.04%, respectively. These are
good two-rank results, but adding more ranks does not continue the gains:
at four ranks, efficiency ranges from 23.08% to 46.36%, and by 64 ranks it is
below 1.7% for every size.

### Computation and communication

[`mpi_time_breakdown_plots.png`](./mpi_time_breakdown_plots.png) shows that
computation does not shrink in proportion to rank count, while communication
usually grows. In this implementation, every rank receives all of `B`, `A`
and the result are distributed by rows, and the output is gathered back to
rank 0. With many ranks, this creates more collective communication and leaves
less matrix work per rank to offset that cost.

The stacked bars show compute plus the sum of the three communication phase
medians; they are an explanatory breakdown, not an exact decomposition of the
median total. In particular, the total is the median of complete repetitions,
while each phase value is independently summarized across repetitions and
ranks. Small differences between the bar sum and reported total are therefore
expected.

## Why four CPU cores do not mean four ranks must be fastest

The results show two ranks as the fastest configuration for all three sizes.
Four ranks are sometimes close, particularly for the largest matrix, but are
not faster in this run. A four-core host sets a practical ceiling on how much
CPU work can happen simultaneously; it does not guarantee that using one rank
per core is optimal. The MPI collectives, synchronization, matrix memory
traffic, and process scheduling all contribute to elapsed time. For `n = 1024`
the work is small enough that these costs quickly dominate; for larger inputs
there is more useful computation, so two ranks obtain a substantial speedup.

Runs with 8–64 ranks use more MPI processes than the four available cores.
Those processes compete for CPU time, and this oversubscription is especially
costly here because the program also communicates matrices among ranks.
Oversubscription is consistent with the rising communication and poor scaling,
although the logs do not isolate its effect from other overheads.

The benchmark does not record CPU model, operating system, MPI implementation,
rank pinning/placement, or system load. These details are needed for exact
reproduction and can affect timing. Also, four repetitions provide only a
small sample: medians reduce sensitivity to one unusually fast or slow run,
but the results should be treated as measurements of this run rather than
universal performance guarantees.

## Conclusion

For this four-core baseline, **two MPI ranks are the best measured setting**
for all tested matrix sizes. They reduce total time by about 36% for
`n = 1024` and about 48% for `n = 2048` and `n = 4096` compared with one rank.
Four ranks retain a speedup for the two larger matrices but do not beat the
two-rank time. Beyond four ranks, extra parallelism is increasingly
counterproductive; at 64 ranks, the largest matrix is only about 1.04× faster
than one rank and the two smaller matrices are slower than one rank.

The central result is that rank count alone does not determine performance.
On this hardware and with this broadcast/scatter/compute/gather design, the
two-rank configuration provides the best balance between parallel work and
communication overhead. The benchmark is also evidence that the 8–64-rank
measurements are oversubscribed runs, not evidence of performance on systems
with that many cores.
