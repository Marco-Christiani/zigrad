# Matmul Benchmark

Compares matrix-multiply performance across implementations, currently used for validating integrations.

## Implementations

| Name | Description |
|------|-------------|
| `zig_naive` | Triple-loop baseline (no optimizations) |
| `blas` | Intel MKL `cblas_sgemm` (requires MKL in SDK, f32 only) |
| `tvm_cpu` | TVM MetaSchedule-tuned kernel (CPU, requires prior tuning) |
| `tvm_gpu` | TVM MetaSchedule-tuned kernel (CUDA) |
| `iree` | IREE-compiled StableHLO |
| `xla_cpu` | XLA/PJRT compiled StableHLO (CPU) |
| `xla_gpu` | XLA/PJRT compiled StableHLO (GPU) |

Shorthands: `all`, `all-cpu`, `all-gpu`.

## Data Types

| Flag | Notes |
|------|-------|
| `--dtype=f32` | Default. All implementations supported. |
| `--dtype=f16` | IEEE half-precision. Support depends on the selected implementation. |

## Enter the development environment

From the Zigrad repository root:

```sh
nix develop --impure
```

The devshell sets `ZG_EXTERNAL_SDK_ROOT`, the TVM runtime paths, and the CPU and
 GPU PJRT plugin paths. A direct Zig workflow outside the devshell must provide
 an equivalent SDK with `-Dsdk=<path>` or `ZG_EXTERNAL_SDK_ROOT`.

## Build

```sh
cd examples/benchmark
zig build
zig build test --summary all
```

## Run

```sh
# Naive baseline only (no external dependencies)
zig build run -- --impls=zig_naive --shapes=128x128x128

# Compare naive vs BLAS (f32)
zig build run -- --impls=zig_naive,blas --shapes=128x128x128,256x256x256

# All CPU implementations
zig build run -- --impls=all-cpu --shapes=512x512x512 --iters=50

# Compare one compiled program through IREE and XLA/PJRT
zig build run -- --impls=iree,xla_cpu --shapes=128x128x128

# All GPU implementations
zig build run -- --impls=all-gpu --shapes=1024x1024x1024

# Everything
zig build run -- --impls=all --shapes=256x256x256
```

The XLA adapters read distinct CPU and GPU plugin paths so both implementations
 can be selected in one run:

```sh
PJRT_CPU_PLUGIN_PATH=/path/to/cpu/plugin.so zig build run -- \
  --impls=xla_cpu --shapes=128x128x128

PJRT_GPU_PLUGIN_PATH=/path/to/gpu/plugin.so zig build run -- \
  --impls=xla_gpu --shapes=128x128x128
```

TVM execution requires a target-specific tuning record. Run tuning from the
 Zigrad repository root before starting the benchmark:

```sh
nix run --impure .#zigrad-dev-cuda -- tvm tune \
  --shape 128x128x128 --trials 64 --trials-per-iter 16 --cpu
```

Use `--gpu` for the CUDA target. Tuning and execution share
 `ZG_CACHE_DIR`, which defaults to `/tmp/zigrad-cache`.

## Output

The benchmark prints median time, GFLOP/s, correctness status, and speedup
 relative to the naive Zig result when that implementation is selected.
