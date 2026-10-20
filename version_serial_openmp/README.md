# Benchmark result: serial baseline vs. parallel Numba version on a 4-core local CPU

This folder contains the benchmark logs and plots comparing the serial implementation (`v0_serial`) against the parallel Numba implementation (`v1_parallel`) for matrix multiplication on sizes 1024, 2048, and 4096.

This experiment was run locally on a machine with 4 CPU cores. That detail is important because the benchmark intentionally tests thread counts up to 8, which is above the available physical core count. In other words, the `p = 8` configuration is effectively oversubscribed on this machine.

## What the benchmark measures

The benchmark records the median execution time over four runs for each matrix size and thread count. From those times, it computes:

- Speedup: `T1 / Tp`
- Efficiency: `Speedup / p`

With ideal scaling, speedup should follow `p` and efficiency should remain at `1.0` (100%). In practice, both fall below ideal values as thread count grows.

## Baseline serial results

The serial version (`v0_serial`) is the reference:

- `n = 1024`: 0.1736 s median
- `n = 2048`: 2.7424 s median
- `n = 4096`: 23.9956 s median

## Parallel results from the logs

### Matrix size n = 1024

- `p = 1`: ~1.06x speedup, 106% efficiency
- `p = 2`: ~1.97x speedup, 98% efficiency
- `p = 4`: ~2.04x speedup, 51% efficiency
- `p = 8`: ~2.05x speedup, 26% efficiency

This is a small workload, so adding threads does not help much after the first few. The runtime is already tiny, and the overhead of thread scheduling and synchronization becomes a large fraction of total cost.

### Matrix size n = 2048

- `p = 1`: ~1.01x speedup, 101% efficiency
- `p = 2`: ~1.61x speedup, 80% efficiency
- `p = 4`: ~2.45x speedup, 61% efficiency
- `p = 8`: ~2.64x speedup, 33% efficiency

This is where the parallel version shows a clear advantage over the serial code. Runtime improves from ~2.74 s to about 1.04 s at `p = 8`, but the efficiency drops because the work is no longer scaling linearly with the extra threads.

### Matrix size n = 4096

- `p = 1`: ~1.02x speedup, 102% efficiency
- `p = 2`: ~1.82x speedup, 91% efficiency
- `p = 4`: ~2.41x speedup, 60% efficiency
- `p = 8`: ~2.25x speedup, 28% efficiency

This is the largest input and therefore the most representative of the real benefit of parallelism. The runtime drops significantly compared to the serial baseline, but the best result is still not at the maximum thread count. On a 4-core CPU, `p = 8` is beyond the physical core limit, so the runtime degrades due to contention and context switching.

## Interpretation of the plots

The file `v0_v1_oversubscribed_plots.png` shows three key trends:

1. Speedup vs. threads
   - Speedup rises as threads increase, but it stays well below the ideal line.
   - For all matrix sizes, the curve bends away from the ideal linear trend as `p` grows.

2. Efficiency vs. threads
   - Efficiency is near ideal at low thread counts, then drops sharply.
   - By `p = 8`, the efficiency is only about 25–33%, which indicates that extra threads are no longer contributing useful work efficiently.

3. Execution time vs. threads
   - Runtime improves from 1 to 2 or 4 threads, but it does not keep declining at 8.
   - For the largest matrix, the 4-thread case is faster than the 8-thread case (`9.96 s` vs `10.67 s`), which is a clear sign of oversubscription.

## Why performance stops scaling

Because this machine has only 4 physical cores, running the benchmark at `p = 8` means the operating system must schedule more threads than there are cores available. This causes:

- CPU contention between threads
- cache thrashing and memory-bandwidth pressure
- extra scheduling and synchronization overhead
- reduced gains for small or medium matrix sizes

This is why the benchmark demonstrates oversubscription rather than perfect parallel scaling.

## Overall conclusion

The parallel Numba version is faster than the serial baseline for medium and large matrix sizes, but it does not scale linearly. The best runtime on this 4-core local CPU is generally reached at `p = 2` or `p = 4`, while `p = 8` is no longer efficient and can even become slower.

The main lesson is that the fastest configuration is not always the one with the largest thread count. On a 4-core machine, using 8 threads oversubscribes the system and reduces the benefit of parallel execution.

## Summary

- Serial baseline is the reference implementation.
- Parallel version helps substantially for larger matrices.
- The best speedup occurs at moderate thread counts, not the maximum.
- On a 4-core CPU, `p = 8` is oversubscribed and therefore underperforms relative to `p = 4`.
- The benchmark confirms that hardware limits and synchronization overhead matter as much as raw thread count.
