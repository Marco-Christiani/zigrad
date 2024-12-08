import os
import argparse
from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some strings.')    
    parser.add_argument('cuda_path', type=str)    
    args = parser.parse_args()
    cuda_path = args.cuda_path

    here = Path(__file__).parent.resolve()
    root = here / Path("src/cuda/cuda_includes.cu")

    roots = {
        "HOST": "host_device.zig",
        "CUDA": "cuda_device.zig",
    }

    if (cuda_path[-1] == '/'):
        cuda_path = cuda_path[0:-1]

    headers = [
        f'#include "{cuda_path}/include/cuda.h"',
        f'#include "{cuda_path}/include/cublas_v2.h"',
        f'#include "{cuda_path}/include/thrust/device_ptr.h"',
        f'#include "{cuda_path}/include/thrust/functional.h"',
        f'#include "{cuda_path}/include/thrust/reduce.h"',
        f'#include "{cuda_path}/include/thrust/inner_product.h"',
        f'#include "{cuda_path}/include/thrust/fill.h"',
        f'#include "{cuda_path}/include/thrust/sequence.h"',
        f'#include "{cuda_path}/include/thrust/transform.h"',
        f'#include "{cuda_path}/include/thrust/tuple.h"',
        f'#include "{cuda_path}/include/thrust/iterator/zip_iterator.h"',
        f'#include "{cuda_path}/include/thrust/iterator/counting_iterator.h"',
        f'#include "{cuda_path}/include/thrust/execution_policy.h"',
        f'#include "{cuda_path}/include/cudnn.h"',
    ]
    
    with open(root, 'w') as file:
        file.write("\n".join(headers))
        
