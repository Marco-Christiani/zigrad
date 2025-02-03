#!/usr/bin/env python3
from pathlib import Path
import argparse
import subprocess
import sys
import os
import json

# root of this path
HERE = Path(__file__).parent.resolve()

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

def remaining_paths(found: dict[str, str], total: list[str]) -> list[str]:
    remaining = []
    for path in total:
        if path not in found:
            remaining.append(path)
    return remaining

def check_path_variable():
    # Get PATH from the environment
    path_env = os.environ.get("PATH", "")
    if not path_env:
        print("PATH environment variable is empty.", file=sys.stderr)
        sys.exit(1)
    
    # Split PATH into directories (using ':' on Linux)
    dirs = path_env.split(os.pathsep)
    
    # Filter directories that contain "cuda" (case-insensitive)
    cuda_dirs = list(set([d for d in dirs if "cuda" in d.lower()]))
    
    # Raise error if not exactly one is found
    if len(cuda_dirs) == 0:
        print("No CUDA directory found in PATH.", file=sys.stderr)
        return None
    if len(cuda_dirs) > 1:
        print("Multiple CUDA directories found in PATH:", file=sys.stderr)
        print(", ".join(cuda_dirs), file=sys.stderr)
        return None
    
    candidate = cuda_dirs[0]
    
    # Split the candidate into path components.
    # Note: if the candidate is an absolute path, the first element may be empty.
    parts = candidate.split(os.path.sep)
    
    # Find the first component that starts with "cuda" (ignoring case)
    cuda_index = None
    for i, part in enumerate(parts):
        if part.lower().startswith("cuda"):
            cuda_index = i
            break
    if cuda_index is None:
        # This is unlikely since we filtered on "cuda" but it is here for safety.
        print(f"Found candidate '{candidate}' does not have a directory component starting with 'cuda'.", file=sys.stderr)
        return None
    
    # Rebuild the path up to and including the cuda directory.
    truncated = os.path.sep.join(parts[:cuda_index+1])
    
    # Ensure the result ends with a trailing slash.
    if not truncated.endswith(os.path.sep):
        truncated += os.path.sep

    return truncated


###################################################################
#### CUDA SEARCH UTILITY ##########################################
###################################################################

def ordered_path_set(full_paths: dict[str, str]) -> list[str]:
    # we have to preserve order otherwise nvcc will
    # complain things are not defined (like size_t).
    return list(set([value[0:max(value.find(key) - 1, 1)] for key,value in full_paths.items()]))


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
    include_paths: list[str],
    library_paths: list[str],
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
        "-o",
        str(here.parent / "src/cuda/libamalgamate.so"),
        str(here.parent / "src/cuda/amalgamate.cu"),
        "-O3",
        compute_arch,
        "--std=c++20",
        "--expt-extended-lambda",
        "--expt-relaxed-constexpr",
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
        "-lcutensor"
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
        print("Compilation: Success...")
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



def get_user_option(options, prompt: str):

    if len(options) == 1:
        return options[0]

    options.sort()

    print(prompt)
    # Display the options numbered starting from 1
    for index, option in enumerate(options):
        print(f"  [{index}] {option}")
    
    while True:
        # Prompt the user for input
        user_input = input("Please choose an option by entering its number: ")
        
        try:
            # Try to convert the input to an integer
            choice = int(user_input)
            
            # Check if the choice is within the valid range
            if 0 <= choice <= len(options):
                return options[choice]
            else:
                print(f"Invalid choice. Please enter a number between 0 and {len(options)}.")
        except ValueError:
            # If conversion fails, the input wasn't numeric
            print("Invalid input. Only numeric inputs are allowed. Please try again.")


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
            confirmed = get_user_option(candidate_paths, f"Select path for: {lib}")
        else:
            print(f"\nNo candidate paths found for '{lib}' using ldconfig.")
            confirmed = input(f"Please enter the library path for '{lib}' manually (or leave blank to skip): ").strip()

            if not confirmed:
                sys.exit(1)

        if confirmed:
            found_paths[lib] = confirmed

        else:
            print(f"Library '{lib}' was not configured.")
    
    return found_paths


def locate_include_paths(header_files):
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
    search_dirs = []
    
    if (path_env := check_path_variable()) is not None:
        search_dirs.append(path_env)

    # pick some obvious candidates to check first
    search_dirs.extend(['/usr/local', "/usr/include", "/opt"])

    found_headers = dict()

    for header in header_files:
        candidate_paths = []        

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
            confirmed = get_user_option(candidate_paths, f"Select path for: {header}")
        else:
            print(f"\nNo candidate paths found for '{header}'.")
            confirmed = input(f"Please enter the path for '{header}' manually (or leave blank to skip): ").strip()

            if not confirmed:
                sys.exit(1)

        if confirmed:
            found_headers[header] = confirmed
        else:
            print(f"Header '{header}' was not configured.")
    
    return found_headers


def check_config() -> tuple[list[str], list[str]]:
    global HERE

    CONFIG_PATH = HERE / "cuda_config.json"

    if not os.path.exists(CONFIG_PATH):
        return None

    with open(CONFIG_PATH, 'r') as file:
        config = json.load(file)

    include_paths = list(set([ value for key,value in config.items() if 'include' in key ]))
    library_paths = list(set([ value for key,value in config.items() if 'library' in key ]))

    return (include_paths, library_paths)
    
    

def main():
    global HERE
    
    parser = argparse.ArgumentParser(description='Process some strings.')    
    parser.add_argument('rebuild', type=str)
    args = parser.parse_args()

    rebuild = args.rebuild == 'y'
    AMALGAMATE_LIBRARY = HERE.parent / "src/cuda/amalgamate.so"

    # we've already built the amalgamate.so and aren't rebuilding
    if not rebuild and os.path.exists(AMALGAMATE_LIBRARY):
        return

    # CHECK IF WE HAVE A CONFIGURED SET OF PATHS #
    if (config := check_config()) is not None:
        print("Compiling amalgamate.so from provided cuda_conifg.json...")
        compile_cuda(config[0], config[1])
        return
        
    LIBRARY_PATHS_CACHE = HERE.parent / "src/cuda/.library_paths.cache"
    INCLUDE_PATHS_CACHE = HERE.parent / "src/cuda/.include_paths.cache"
        
    # Step 1: Check if nvcc is installed and working.
    check_nvcc()

    if not rebuild and os.path.exists(LIBRARY_PATHS_CACHE):
        with open(LIBRARY_PATHS_CACHE, "r") as file:
            library_paths = json.load(file)
    else:
        print("")
        print("############################################################")
        print("#### LOCATING LIBRARY INSTALLATION PATHS ###################")
        library_paths = locate_library_paths(library_paths_to_check)
        with open(LIBRARY_PATHS_CACHE, "w") as file:
            file.write(json.dumps(library_paths))
        
    if not rebuild and os.path.exists(INCLUDE_PATHS_CACHE):
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
    with open(HERE.parent / "src/cuda/cuda_includes.cu", 'w') as file:
        file.write("\n".join((f'#include "{h}"' for h in include_paths.values())))

    compile_cuda(ordered_path_set(include_paths), ordered_path_set(library_paths))

        
if __name__ == '__main__':
    main()
    
