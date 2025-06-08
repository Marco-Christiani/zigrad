cmake_minimum_required(VERSION 3.18)
project(amalgamate LANGUAGES CUDA)

# Set the NVCC compiler flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --std=c++20 --expt-extended-lambda --expt-relaxed-constexpr --compiler-options -fPIC")

# Create a shared library from the single CUDA source file
add_library(amalgamate SHARED amalgamate.cu)

# Optionally, explicitly set the output file name (amalgamate.so)
set_target_properties(amalgamate PROPERTIES OUTPUT_NAME "amalgamate")
set_target_properties(amalgamate PROPERTIES CUDA_ARCHITECTURES native)

# Link the required CUDA libraries
target_link_libraries(amalgamate PRIVATE cuda cudart cublas cutensor cudnn nvrtc)
