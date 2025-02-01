#!/usr/bin/env python3
from pathlib import Path
import argparse
import subprocess
import sys
import os
import json

def ends_with(path: Path, tail: str) -> bool:
    return str(path).endswith(tail)

def get_cuda_compute() -> str:    
    """
    Uses nvidia-smi to query each GPU’s index, name, and compute capability.
    Returns a list of dictionaries with keys 'index', 'name', and 'compute_cap'.
    """
    try:
        # Query GPUs for index, name, and compute capability in CSV format.
        # (Note: the 'compute_cap' query field is supported in newer driver versions.)
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,name,compute_cap",
            "--format=csv,noheader"
        ]
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
        parts = [p.strip() for p in line.split(',')]
        if len(parts) >= 3:
            gpu_info = {
                "index": parts[0],
                "name": parts[1],
                # transform x.y -> sm_xy
                "compute_cap": f'sm_{parts[2].replace('.', '')}',
            }
            gpu_list.append(gpu_info)
    return gpu_list

def choose_gpu(gpu_list):
    """
    If multiple GPUs are available, prints the list and asks the user to select one.
    If only one GPU is found, it is automatically selected.
    Returns the chosen GPU dictionary.
    """
    if not gpu_list:
        print("No NVIDIA GPUs found.")
        sys.exit(1)

    if len(gpu_list) == 1:
        gpu = gpu_list[0]
        print("Only one GPU found:")
        print(f"  GPU {gpu['index']}: {gpu['name']} (Compute Capability: {gpu['compute_cap']})")
        return gpu['compute_cap']

    # More than one GPU found: display the list and prompt for a selection.
    print("Multiple NVIDIA GPUs found:")
    for gpu in gpu_list:
        print(f"  GPU {gpu['index']}: {gpu['name']} (Compute Capability: {gpu['compute_cap']})")

    while True:
        selection = input("Enter the GPU index to select: ").strip()
        # Check if the entered index matches one of the GPUs.
        for gpu in gpu_list:
            if gpu["index"] == selection:
                return gpu['compute_cap']
        print("Invalid selection. Please try again.")


def compile_cuda(
    include_paths: set[str],
    library_paths: set[str],
) -> None:
    """
    Compile code for CUDA amalgamate library.
    """

    #const gpu_architecture = std.mem.join(b.allocator, "", &.{ "--gpu-architecture=", gpu_arch }) catch unreachable;
    compute_arch = f'--gpu-architecture={choose_gpu(get_cuda_compute())}'

    here = Path(__file__).parent.resolve()

    nvcc_args = [
        "nvcc",
        "--shared",
        "-allow-unsupported-compiler",
        "-ccbin",
        "/usr/bin/gcc",
        "-o",
        str(here.parent / "src/cuda/libamalgamate.so"),
        str(here.parent / "src/cuda/amalgamate.cu"),
        "-O3",
        compute_arch,
        "--expt-extended-lambda",
        "--compiler-options",
        "-fPIC",
    ]

    nvcc_args.extend([f'-I{path}' for path in include_paths ])
    nvcc_args.extend([f'-L{path}' for path in library_paths ])

    nvcc_args.extend([
        "-lcudart",
        "-lcuda",
        "-lcublas",
        "-lcudnn",
    ])

    print(nvcc_args)

    try:
        print("Compiling CUDA amalgamate.so...")
        # Run nvcc --version and capture its output.
        result = subprocess.run(
            nvcc_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        print(result.stdout)
    except FileNotFoundError:
        print("\nError: nvcc was not found. Please ensure that the NVIDIA CUDA toolkit is installed and nvcc is in your PATH.", end='\n\n')
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print("\nError: nvcc did not run correctly. Details:", end='\n\n')
        print(e.stderr)
        sys.exit(1)


def check_nvcc():
    """
    Checks if nvcc is installed and working by executing 'nvcc --version'.
    Exits the script if nvcc is not found or not working.
    """
    try:
        # Run nvcc --version and capture its output.
        result = subprocess.run(
            ["nvcc", "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        print("\nnvcc is installed. Version info:", end='\n\n')
        print(result.stdout)
    except FileNotFoundError:
        print("\nError: nvcc was not found. Please ensure that the NVIDIA CUDA toolkit is installed and nvcc is in your PATH.", end='\n\n')
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print("\nError: nvcc did not run correctly. Details:", end='\n\n')
        print(e.stderr)
        sys.exit(1)


def locate_library_paths(library_names):
    """
    Attempts to locate the shared library files for each library in the list using ldconfig.
    For each library, candidate paths are presented to the user for confirmation.
    
    Parameters:
        library_names (list of str): Names of the libraries (e.g., "libcudart.so") to locate.
        
    Returns:
        dict: A mapping of library names to confirmed paths.
    """
    found_paths = {}

    try:
        # Run ldconfig to get the list of known shared libraries.
        ldconfig = subprocess.run(
            ["ldconfig", "-p"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
    except subprocess.CalledProcessError as e:
        print("Error: Failed to run ldconfig. You might need to run this script with elevated permissions.")
        return found_paths

    # Split the output into lines for processing.
    ldconfig_lines = ldconfig.stdout.splitlines()

    # Process each library name.
    for lib in library_names:
        candidate_paths = []

        # Search each line for the library name.
        for line in ldconfig_lines:
            # Search exact for ending - controls which version of a library we are looking for
            if ends_with(line, lib):
                # ldconfig output lines are typically formatted like:
                # "    libcudart.so.10.1 (libc6,x86-64) => /usr/local/cuda-10.1/targets/x86_64-linux/lib/libcudart.so.10.1"
                if "=>" in line:
                    parts = line.split("=>")
                    if len(parts) > 1:
                        candidate_paths.append(parts[1].strip())

        if candidate_paths:
            print(f"\nFound candidate paths for '{lib}':")
            confirmed = None
            # Ask the user to confirm one of the candidate paths.
            for path in candidate_paths:
                response = input(f"Use this library path for '{lib}'? {path} (y/n): ").strip().lower()
                if response == 'y':
                    confirmed = path
                    break
            # If no candidate was accepted, ask for manual input.
            if confirmed is None:
                manual = input(f"No candidate accepted for '{lib}'. Please enter the library path manually (or leave blank to skip): ").strip()
                if manual:
                    confirmed = manual
        else:
            print(f"\nNo candidate paths found for '{lib}' using ldconfig.")
            confirmed = input(f"Please enter the library path for '{lib}' manually (or leave blank to skip): ").strip()
            if not confirmed:
                confirmed = None

        if confirmed:
            found_paths[lib] = confirmed
        else:
            print(f"Library '{lib}' was not configured.")
    
    return found_paths


def locate_include_paths(header_files, search_dirs=None):
    """
    Attempts to locate header (.h) files for each header in the list by recursively scanning
    through common include directories.
    
    This updated version allows specifying a partial or relative path. For example,
    a header specified as "thrust/functional.h" will match files whose relative path
    (from the include directory) ends with "thrust/functional.h".
    
    Parameters:
        header_files (list of str): Names (or relative paths) of the header files to locate.
        search_dirs (list of str, optional): Directories to search in. If None, defaults to
                                             ['/usr/include', '/usr/local/include', '/opt'].
    
    Returns:
        dict: A mapping of header file names (as provided) to confirmed full file paths.
    """
    if search_dirs is None:
        # These are common ubuntu directories
        search_dirs = ["/usr/include", "/usr/local/", "/opt"]

    found_headers = {}

    for header in header_files:
        candidate_paths = []
        print(f"\nSearching for header file '{header}' in directories: {', '.join(search_dirs)}")
        
        # Walk through each provided search directory.
        for base in search_dirs:
            for root, dirs, files in os.walk(base):
                for f in files:
                    full_path = os.path.join(root, f)
                    # Compute the relative path from the search directory
                    rel_path = os.path.normpath(os.path.relpath(full_path, base))
                    normalized_header = os.path.normpath(header)
                    
                    # If the header name includes a directory separator, look for files that
                    # end with the provided relative path (e.g., "thrust/functional.h").
                    if os.path.sep in normalized_header:
                        if rel_path.endswith(normalized_header):
                            candidate_paths.append(full_path)
                    else:
                        # Otherwise, check if the filename exactly matches.
                        if f == header:
                            candidate_paths.append(full_path)
        
        # Remove duplicate candidates.
        candidate_paths = list(set(candidate_paths))
        
        if candidate_paths:
            print(f"\nFound candidate paths for '{header}':")
            confirmed = None
            # Ask the user to confirm one of the candidate paths.
            for candidate in candidate_paths:
                response = input(f"Use this header path for '{header}'? {candidate} (y/n): ").strip().lower()
                if response == 'y':
                    confirmed = candidate
                    break
            if confirmed is None:
                manual = input(f"No candidate accepted for '{header}'. Please enter the header path manually (or leave blank to skip): ").strip()
                confirmed = manual if manual else None
        else:
            print(f"\nNo candidate paths found for '{header}' using standard include directories.")
            confirmed = input(f"Please enter the header path for '{header}' manually (or leave blank to skip): ").strip() or None

        if confirmed:
            found_headers[header] = confirmed
        else:
            print(f"Header '{header}' was not configured.")
    
    return found_headers


def minimal_path_set(headers: dict[str, str]) -> list[str]:    
    return set((value[0:max(value.find(key) - 1, 1)] for key,value in headers.items()))


library_paths_to_check = [
    "libcudart.so",
    "libcublas.so", 
    "libcutensor.so",
    "libcudnn.so",
    "libnvrtc.so",
]

include_paths_to_check = [
    "cuda.h",
    "cublas_v2.h",
    "thrust/device_ptr.h",
    "thrust/functional.h",
    "thrust/reduce.h",
    "thrust/inner_product.h",
    "thrust/fill.h",
    "thrust/sequence.h",
    "thrust/transform.h",
    "thrust/copy.h",
    "thrust/tuple.h",
    "thrust/iterator/zip_iterator.h",
    "thrust/iterator/counting_iterator.h",
    "thrust/execution_policy.h",
    "thrust/random.h",
    "cudnn.h",
    "cutensor.h",
]

def main():
    parser = argparse.ArgumentParser(description='Process some strings.')    
    parser.add_argument('--rebuild', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    here = Path(__file__).parent.resolve()
    LIBRARY_PATHS_CACHE = here.parent / "src/cuda/.library_paths.cache"
    INCLUDE_PATHS_CACHE = here.parent / "src/cuda/.include_paths.cache"
    AMALGAMATE_LIBRARY = here.parent / "src/cuda/amalgamate.so"

    # probably the first time building...
    if not os.path.exists(AMALGAMATE_LIBRARY):
        print("")
        print("############################################################")
        print("#### LOCATING CUDA INSTALLATION PATHS (THIS IS A PAIN) #####")
        print("""
        This is an alpha version of this utility and probably sucks.
              
        We will try to locate the required files to install CUDA.
        This will create a "cuda_paths.zig" file that will include
        the minimum set of paths required to install CUDA. You can
        edit that file whenever you want. If you have already done
        this, proceeding will overwrite that file.
        """)

        response = input(f"Do you wish to continue? (y/n): ").strip().lower()

        if response != 'y':
            return
    
    # Step 1: Check if nvcc is installed and working.
    check_nvcc()


    if not args.rebuild and os.path.exists(LIBRARY_PATHS_CACHE):
        with open(LIBRARY_PATHS_CACHE, "r") as file:
            library_paths = json.load(file)
    else:
        print("")
        print("############################################################")
        print("#### LOCATING LIBRARY INSTALLATION PATHS ###################")
        library_paths = locate_library_paths(library_paths_to_check)
        with open(LIBRARY_PATHS_CACHE, "w") as file:
            file.write(json.dumps(library_paths))
        
    if not args.rebuild and os.path.exists(INCLUDE_PATHS_CACHE):
        with open(INCLUDE_PATHS_CACHE, "r") as file:
            include_paths = json.load(file)
    else:
        print("")
        print("############################################################")
        print("#### LOCATING INCLUDE INSTALLATION PATHS ###################")
        include_paths = locate_include_paths(include_paths_to_check)
        with open(INCLUDE_PATHS_CACHE, "w") as file:
            file.write(json.dumps(include_paths))

    ### GENERATE CUDA INCLUDES FROM LOCATED INCLUDE PATHS ###

    print("\nWriting includes to 'cuda_includes.cu'\n")
    with open(here / "cuda_includes.cu", 'w') as file:
        file.write("\n".join((f'#include "{h}"' for h in include_paths.values())))

    if args.rebuild or not os.path.exists(AMALGAMATE_LIBRARY):
        compile_cuda(
            minimal_path_set(include_paths),
            minimal_path_set(library_paths),
        )
    
        
if __name__ == '__main__':
    main()
