#!/usr/bin/env python3
import contextlib
from pathlib import Path
import argparse
import subprocess
import sys
import os


def get_all_cuda_devices():
    """
    Uses nvidia-smi to query each GPU’s index, name, and compute capability.
    Returns a list of dictionaries with keys 'index', 'name', and 'compute_cap'.
    """
    try:
        # Query GPUs for index, name, and compute capability in CSV format.
        # (Note: the 'compute_cap' query field is supported in newer driver versions.)
        cmd = ["nvidia-smi", "--query-gpu=index,name,compute_cap", "--format=csv,noheader"]
        output = subprocess.check_output(cmd, universal_newlines=True)
    except FileNotFoundError:
        print("Error: nvidia-smi not found. Make sure NVIDIA drivers are installed and nvidia-smi is in your PATH.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print("Error: nvidia-smi returned an error:", e)
        sys.exit(1)

    gpu_list = []
    for line in output.strip().splitlines():
        # Each line should be in the form: index, name, compute_cap
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            gpu_info = {
                "index": parts[0],
                "name": parts[1],
                # transform x.y -> sm_xy
                "compute_cap": f"sm_{parts[2].replace('.', '')}",
            }
            gpu_list.append(gpu_info)

    return gpu_list


def get_cuda_compute() -> str:
    """
    If multiple GPUs are available, prints the list and asks the user to select one.
    If only one GPU is found, it is automatically selected.
    Returns the chosen GPU dictionary.
    """
    gpu_list = get_all_cuda_devices()

    if not gpu_list:
        print("No NVIDIA GPUs found.")
        sys.exit(1)
    elif len(gpu_list) == 1:
        gpu = gpu_list[0]
        print("Only one GPU found:")
        print(f"  GPU {gpu['index']}: {gpu['name']} (Compute Capability: {gpu['compute_cap']})")
        return gpu["compute_cap"]
    else:
        g0 = gpu_list[0]
        for g1 in gpu_list[1:]:
            if g0["compute_cap"] != g1["compute_cap"]:
                print("Incompatible compute types found. Pass gpu-arch flag to zig build (ex. -Dcuda_arch=sm_89)")
                sys.exit(1)

        return g0["compute_cap"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cuda_rebuild", type=str)
    args = parser.parse_args()
    cuda_rebuild = args.cuda_rebuild == "y"

    HERE = Path(__file__).parent.resolve()
    CUDA_SRC = HERE.parent / "src" / "cuda"

    # we've already built amalgamate.so and aren't rebuilding
    if not cuda_rebuild and os.path.exists(CUDA_SRC / "amalgamate.so"):
        return

    if not os.path.exists(CUDA_SRC / "build"):
        os.mkdir(CUDA_SRC / "build")

    # with contextlib.chdir(CUDA_SRC / "build"):
    os.chdir(str(CUDA_SRC / "build"))
    if cuda_rebuild or not os.path.exists(CUDA_SRC / "build" / "Makefile"):
        subprocess.run(["cmake", ".."], check=True)

    subprocess.run(["cmake", "--build", "."], check=True)


if __name__ == "__main__":
    main()
