#!/usr/bin/env python3
"""TVM MetaSchedule Autotuning Script

Usage:
    python scripts/tvm_autotune.py matmul --shape 1024x1024x1024 --target llvm --trials 64
    python scripts/tvm_autotune.py matmul --shape 1024x1024x1024 --target cuda --trials 128

Outputs:
    artifacts/tvm_cache/{op}_{shape}_{target}.json  - MetaSchedule database
    artifacts/tvm_cache/{op}_{shape}_{target}.so    - Compiled tuned module
"""

import argparse
import sys
import time
from pathlib import Path

from tvm.script import tir as T

import tvm
from tvm import meta_schedule as ms


def create_matmul_tir(M: int, N: int, K: int) -> tvm.ir.IRModule:
    """Create a TIR function for 2D matmul: C[M,N] = A[M,K] @ B[K,N]

    This uses TVM Script (@tir.prim_func) to define the computation at the TIR level,
      which is what MetaSchedule operates on.
    """

    @T.prim_func
    def matmul(
        A: T.Buffer((M, K), "float32"),
        B: T.Buffer((K, N), "float32"),
        C: T.Buffer((M, N), "float32"),
    ):
        for i, j, k in T.grid(M, N, K):
            with T.block("matmul"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = 0.0
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

    return tvm.IRModule.from_expr(matmul.with_attr("global_symbol", "main"))


def tune_and_build_matmul(
    M: int,
    N: int,
    K: int,
    target: str,
    max_trials: int,
    work_dir: Path,
    output_path: Path,
) -> None:
    """Tune and build optimized matmul using MetaSchedule pass-based API.

    Referenced the official TVM pattern from customize_opt.py tutorial.
    """
    # auto-detect target attributes if not specified
    if target.startswith("llvm"):
        if "num-cores" not in target:
            import multiprocessing

            num_cores = multiprocessing.cpu_count()
            target = f"{target} -num-cores {num_cores}"
            print(f"Auto-detected {num_cores} CPU cores")
    elif target.startswith("cuda"):
        # add required CUDA attributes if not present
        # RTX 3080 Ti (sm_86) specs: 1024 threads/block, 48KB shared memory
        if "arch" not in target:
            target = f"{target} -arch=sm_86"
        if "max_threads_per_block" not in target:
            target = f"{target} -max_threads_per_block=1024"
        if "max_shared_memory_per_block" not in target:
            target = f"{target} -max_shared_memory_per_block=49152"  # 48KB
        if "max_num_threads" not in target:
            target = f"{target} -max_num_threads=1024"
        if "thread_warp_size" not in target:
            target = f"{target} -thread_warp_size=32"
        print(f"CUDA target with required attributes: {target}")

    print(f"\n{'=' * 60}")
    print("TVM MetaSchedule Autotuning & Build")
    print(f"{'=' * 60}")
    print("Operation: matmul")
    print(f"Shape: M={M}, N={N}, K={K}")
    print(f"Target: {target}")
    print(f"Max trials: {max_trials}")
    print(f"Work dir: {work_dir}")
    print(f"Output: {output_path}")
    print(f"{'=' * 60}\n")

    # create TIR function
    mod = create_matmul_tir(M, N, K)
    print("Input TIR Module:")
    print(mod)
    print()

    # parse target
    tvm_target = tvm.target.Target(target)

    work_dir.mkdir(parents=True, exist_ok=True)

    # use direct TIR tuning api (not Relax passes - those are for Relax+TIR)
    print(f"Starting tuning with {max_trials} trials...")
    start_time = time.time()

    database = ms.tune_tir(
        mod=mod,
        target=tvm_target,
        work_dir=str(work_dir),
        max_trials_global=max_trials,
        num_trials_per_iter=max_trials,  # All trials in one batch for small counts
    )

    elapsed = time.time() - start_time
    print(f"\nTuning completed in {elapsed:.2f}s")

    print("Applying best schedule from database...")

    # get all tuning records and find the best one
    records = database.get_all_tuning_records()
    if records:
        # run_secs are TVM FloatImm objects, convert to float
        def avg_time(r):
            return sum(float(t) for t in r.run_secs) / len(r.run_secs)

        best_record = min(records, key=avg_time)
        best_time_us = avg_time(best_record) * 1e6
        print(f"  Found {len(records)} records, best: {best_time_us:.2f} µs")

        # apply the trace to the module
        sch = tvm.tir.Schedule(mod)
        best_record.trace.apply_to_schedule(sch, remove_postproc=False)
        optimized_mod = sch.mod
        print("\N{CHECK MARK} Best schedule applied via trace replay")
    else:
        print("WARNING: No tuning records found, using default schedule")
        optimized_mod = mod

    print("\nOptimized TIR Module:")
    print(optimized_mod)
    print()

    print(f"Building optimized module to {output_path}...")
    with tvm.transform.PassContext(opt_level=3):
        compiled_lib = tvm.build(optimized_mod, target=tvm_target)

    # export to .so
    output_path.parent.mkdir(parents=True, exist_ok=True)
    compiled_lib.export_library(str(output_path))

    print(f"\N{CHECK MARK} Compiled module saved to {output_path}")
    print(f"  Size: {output_path.stat().st_size / 1024:.2f} KB")


# NOTE: compile_best_schedule removed - now using pass-based API in tune_and_build_matmul


def verify_tuned_module(
    module_path: Path,
    M: int,
    N: int,
    K: int,
    device_type: str = "cpu",
) -> None:
    """Load the tuned module and verify it produces correct results."""
    import numpy as np

    print(f"\nVerifying tuned module on {device_type}...")
    loaded_lib = tvm.runtime.load_module(str(module_path))

    np.random.seed(0)
    a_np = np.random.rand(M, K).astype("float32")
    b_np = np.random.rand(K, N).astype("float32")
    c_np = np.zeros((M, N), dtype="float32")

    # reference
    c_ref = a_np @ b_np

    # create TVM tensors (tvm.runtime.tensor instead of tvm.nd.array, did I see tvm.nd.array somewhere?)
    dev = tvm.device(device_type, 0)
    a_tvm = tvm.runtime.tensor(a_np, dev)
    b_tvm = tvm.runtime.tensor(b_np, dev)
    c_tvm = tvm.runtime.tensor(c_np, dev)

    # execute tuned kernel (function is named "main" in TIR)
    loaded_lib["main"](a_tvm, b_tvm, c_tvm)

    np.testing.assert_allclose(c_tvm.numpy(), c_ref, rtol=1e-5, atol=1e-5)

    print("\N{CHECK MARK} Verification passed: tuned kernel produces correct results")


def benchmark_module(
    module_path: Path,
    M: int,
    N: int,
    K: int,
    device_type: str = "cpu",
    num_repeats: int = 100,
) -> float:
    """Benchmark tuned module and return average execution ms."""
    import numpy as np

    print(f"\nBenchmarking tuned module on {device_type} ({num_repeats} iterations)...")

    loaded_lib = tvm.runtime.load_module(str(module_path))

    # set up test data
    np.random.seed(0)
    a_np = np.random.rand(M, K).astype("float32")
    b_np = np.random.rand(K, N).astype("float32")
    c_np = np.zeros((M, N), dtype="float32")

    dev = tvm.device(device_type, 0)
    a_tvm = tvm.runtime.tensor(a_np, dev)
    b_tvm = tvm.runtime.tensor(b_np, dev)
    c_tvm = tvm.runtime.tensor(c_np, dev)

    # warmup
    for _ in range(10):
        loaded_lib["main"](a_tvm, b_tvm, c_tvm)

    # benchmark
    import time

    times = []
    for _ in range(num_repeats):
        start = time.time()
        loaded_lib["main"](a_tvm, b_tvm, c_tvm)
        dev.sync()
        times.append(time.time() - start)

    avg_time_ms = (sum(times) / len(times)) * 1000
    print(f"  Average time: {avg_time_ms:.3f} ms")
    print(f"  Min time: {min(times) * 1000:.3f} ms")
    print(f"  Max time: {max(times) * 1000:.3f} ms")

    return avg_time_ms


def parse_shape(shape_str: str) -> tuple[int, int, int]:
    """Parse shape string like '1024x1024x1024' into (M, N, K)"""
    parts = shape_str.split("x")
    if len(parts) != 3:
        raise ValueError(f"Shape must be MxNxK, got: {shape_str}")
    return tuple(int(p) for p in parts)


def main():
    parser = argparse.ArgumentParser(
        description="TVM MetaSchedule autotuning for matrix operations"
    )
    parser.add_argument(
        "operation",
        choices=["matmul"],
        help="Operation to tune",
    )
    parser.add_argument(
        "--shape",
        type=str,
        default="1024x1024x1024",
        help="Operation shape (format: MxNxK for matmul)",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="llvm",
        help="TVM target (llvm, llvm -mcpu=native, cuda, etc.)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=64,
        help="Number of tuning trials",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("artifacts/tvm_cache"),
        help="Directory to store tuning cache and compiled modules",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify the tuned module produces correct results",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Benchmark the tuned module after compilation",
    )

    args = parser.parse_args()

    M, N, K = parse_shape(args.shape)

    # construct output filenames
    target_name = args.target.replace(" ", "_").replace("-", "_")
    op_signature = f"{args.operation}_{M}x{N}x{K}_{target_name}"

    work_dir = args.cache_dir / f"{op_signature}_work"
    module_path = args.cache_dir / f"{op_signature}.so"

    # derive device type from target
    device_type = "cuda" if args.target.startswith("cuda") else "cpu"

    # tune and build
    if args.operation == "matmul":
        tune_and_build_matmul(
            M=M,
            N=N,
            K=K,
            target=args.target,
            max_trials=args.trials,
            work_dir=work_dir,
            output_path=module_path,
        )
    else:
        print(f"Operation {args.operation} not implemented yet")
        return 1

    if args.verify:
        verify_tuned_module(module_path, M, N, K, device_type)

    if args.benchmark:
        benchmark_module(module_path, M, N, K, device_type)

    print(f"\n{'=' * 60}")
    print("Tuning Complete.")
    print(f"{'=' * 60}")
    print(f"Compiled module: {module_path}")
    print(f"Work directory: {work_dir}")
    print("\nTo use this module in Zig:")
    print(f"  zig build run -- tvm-vec-add {module_path}")  # NOTE: this is prone to rot
    print(f"{'=' * 60}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
