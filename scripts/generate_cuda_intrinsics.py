#!/usr/bin/env python3
"""Generate pre-serialized CUDA tensor intrinsics for Zig to load at runtime.

Extracts all CUDA tensor intrinsics registered by TVM's Python module and serializes
them to JSON files so we dont need python at runtime.
"""

import contextlib
import tvm
from tvm.tir import TensorIntrin
from tvm.tir.tensor_intrin import cuda  # Triggers intrinsic registration
import json
import os
from pathlib import Path


def main():
    output_dir = Path("artifacts/cuda_intrinsics")
    output_dir.mkdir(parents=True, exist_ok=True)

    # get all CUDA intrinsic names from the module
    intrinsic_names = set()

    # 1. Get constants defined in cuda module
    for attr in dir(cuda):
        if attr.endswith("_INTRIN") and not attr.startswith("_"):
            value = getattr(cuda, attr)
            if isinstance(value, str):
                intrinsic_names.add(value)

    # 2. Search for dynamically registered MMA intrinsics
    # these dont have constants but are registered at import time
    shapes = [(16, 8, 8), (16, 8, 16), (8, 8, 16), (16, 16, 16), (8, 8, 32)]
    dtypes = ["f16", "f32", "i8", "i32", "s8", "s32", "s4"]
    prefixes = ["mma_init", "mma_fill", "mma_load", "mma_store", "mma_sync"]
    suffixes = [
        "",
        "_A",
        "_B",
        "_C",
        "_A_shared",
        "_B_shared",
        "_C_shared",
        "_A_shared_dyn",
        "_B_shared_dyn",
        "_global",
        "_shared_dyn",
    ]

    # single-dtype pattern (for init, fill, load, store)
    for m, n, k in shapes:
        for dtype in dtypes:
            for prefix in ["mma_init", "mma_fill", "mma_load", "mma_store"]:
                for suffix in suffixes:
                    name = f"{prefix}_m{m}n{n}k{k}_{dtype}{suffix}"
                    with contextlib.suppress(Exception):
                        TensorIntrin.get(name)
                        intrinsic_names.add(name)

    # three-dtype pattern for mma_sync (input_A, input_B, output_C types)
    # common combinations
    sync_dtype_combos = [
        "f16f16f16",
        "f16f16f32",
        "s8s8s32",
        "s4s4s32",
        "i8i8i32",
        "i8i8i32",
    ]
    for m, n, k in shapes:
        for dtype_combo in sync_dtype_combos:
            for suffix in suffixes:
                name = f"mma_sync_m{m}n{n}k{k}_{dtype_combo}{suffix}"
                with contextlib.suppress(Exception):
                    TensorIntrin.get(name)
                    intrinsic_names.add(name)

    intrinsic_names = sorted(intrinsic_names)

    print(f"Generating {len(intrinsic_names)} CUDA tensor intrinsics...")
    print(f"Output directory: {output_dir}")
    print()

    generated = []
    failed = []

    for intrinsic_id in intrinsic_names:
        try:
            # get the registered intrinsic (already registered by import)
            intrin = TensorIntrin.get(intrinsic_id)

            # extract desc and impl PrimFuncs
            desc = intrin.desc
            impl = intrin.impl

            # serialize to json w tvm
            desc_json = tvm.ir.save_json(desc)
            impl_json = tvm.ir.save_json(impl)

            intrinsic_data = {
                "name": intrinsic_id,
                "desc": desc_json,
                "impl": impl_json,
            }

            output_file = output_dir / f"{intrinsic_id}.json"
            output_file.write_text(json.dumps(intrinsic_data, indent=2))

            size_kb = (len(desc_json) + len(impl_json)) / 1024
            generated.append(intrinsic_id)
            print(f"\N{CHECK MARK} {intrinsic_id:40s} {size_kb:6.1f} KB")

        except Exception as e:
            failed.append((intrinsic_id, str(e)))
            print(f"\N{BALLOT X} {intrinsic_id:40s} FAILED: {e}")

    total_size = sum(
        os.path.getsize(output_dir / f)
        for f in os.listdir(output_dir)
        if f.endswith(".json")
    )

    print()
    print("=" * 60)
    print(
        f"\N{CHECK MARK} Generated {len(generated)}/{len(intrinsic_names)} intrinsics"
    )
    print(
        f"  Total size: {total_size / 1024:.1f} KB ({total_size / (1024 * 1024):.2f} MB)"
    )
    print(f"  Output: {output_dir}")

    if failed:
        print(f"\n\N{BALLOT X} Failed: {len(failed)}")
        for name, error in failed:
            print(f"  - {name}: {error}")
        return 1

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
