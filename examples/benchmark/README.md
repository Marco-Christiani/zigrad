# Matmul Benchmark

Compares matrix-multiply performance across implementations, currently used for validating integrations.

## Implementations

| Name | Description |
|------|-------------|
| `zig_naive` | Triple-loop baseline (no optimizations) |
| `blas` | Intel MKL `cblas_sgemm` (requires MKL in SDK, f32 only) |
| `tvm_cpu` | TVM MetaSchedule-tuned kernel (CPU, requires prior tuning) |
| `tvm_gpu` | TVM MetaSchedule-tuned kernel (CUDA) |
| `xla_cpu` | XLA/PJRT compiled StableHLO (CPU) |
| `xla_gpu` | XLA/PJRT compiled StableHLO (GPU) |

Shorthands: `all`, `all-cpu`, `all-gpu`.

## Data Types

| Flag | Notes |
|------|-------|
| `--dtype=f32` | Default. All implementations supported. |
| `--dtype=f16` | IEEE half-precision. BLAS returns error (MKL has no f16 sgemm). |

## Prerequisites

- Zigrad devshell or otherwise valid SDK bundle with `ZG_EXTERNAL_SDK_ROOT` pointing to SDK with MLIR, TVM, and/or MKL headers
- For TVM benchmarks: populated cache (use `zigrad tvm-tune`)
- For XLA benchmarks: `PJRT_CPU_PLUGIN_PATH` or `PJRT_GPU_PLUGIN_PATH` (handled automatically when using devshell)

## Build

```sh
cd examples/benchmark
zig build
# If needed, explicitly specify  -Dsdk=<path-to-sdk-root> (must have been built with tvm, mkl, xla, etc support)
```

## Run

```sh
# Naive baseline only (no external dependencies)
zig build run -- --impls=zig_naive --shapes=128x128x128

# Compare naive vs BLAS (f32)
zig build run -- --impls=zig_naive,blas --shapes=128x128x128,256x256x256

# Run tuning as desired, can be incremental
zig build run -- tune --impls=tvm_cpu --shapes=128x128x128 --trials=64 --trials-per-iter=16

# All CPU implementations
zig build run -- --impls=all-cpu --shapes=512x512x512 --iters=50

# All GPU implementations
zig build run -- --impls=all-gpu --shapes=1024x1024x1024

# Everything
zig build run -- --impls=all --shapes=256x256x256
```

## Output

Prints a table per shape with median time (us), GFLOP/s, and speedup vs naive baseline, followed by a summary of best-performing implementations.
