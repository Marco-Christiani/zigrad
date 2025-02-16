cmake_minimum_required(VERSION 3.18)
project(amalgamate LANGUAGES CUDA)

# Allow the user to specify the GPU architecture (e.g., sm_89) when running cmake:
#   cmake -DGPU_ARCHITECTURE=sm_89 ..
set(GPU_ARCHITECTURE "" CACHE STRING "Target GPU architecture (e.g., sm_89) for NVCC's --gpu-architecture flag.")

# If the user provided a GPU architecture, append the flag to NVCC flags.
if(GPU_ARCHITECTURE)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --gpu-architecture=${GPU_ARCHITECTURE}")
endif()

# Set the NVCC compiler flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --gpu-architecture=${GPU_ARCHITECTURE} --std=c++20 --expt-extended-lambda --expt-relaxed-constexpr --compiler-options -fPIC")

# Create a shared library from the single CUDA source file
add_library(amalgamate SHARED amalgamate.cu)

# Optionally, explicitly set the output file name (amalgamate.so)
set_target_properties(amalgamate PROPERTIES OUTPUT_NAME "amalgamate")

# Link the required CUDA libraries
target_link_libraries(amalgamate PRIVATE cudart cublas cutensor cudnn nvrtc)
