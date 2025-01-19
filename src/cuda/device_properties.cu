#ifndef __ZIG_DEVICE_PROPERTIES__
#define __ZIG_DEVICE_PROPERTIES__

#include <stdio.h>
#include "cuda_helpers.cu"
#include <cuda_runtime.h>

typedef struct {
  int major;
  int minor;
} ComputeVersion;

typedef struct {
  int bus_id;
  int device_id;
} PCIE;

typedef struct {
  size_t total_global;
  size_t total_const;
  size_t shared_per_block;
  size_t shared_per_multiprocessor;
} DeviceMemory;

typedef struct {
  int per_block;
  int per_multiprocessor;
} DeviceRegisters;

typedef struct {
  int x;
  int y;
  int z;
} GridSizes;

typedef struct {
  int x;
  int y;
  int z;
} ThreadDimensions;

struct DeviceProperties {

  DeviceProperties(cudaDeviceProp prop) :
    multi_processor_count{ prop.multiProcessorCount },
    max_threads_per_multi_processor{ prop.maxThreadsPerMultiProcessor },
    max_threads_per_block{ prop.maxThreadsPerBlock },
    pcie{
      .bus_id = prop.pciBusID,
      .device_id = prop.pciDeviceID,
    },
    compute_version{
      .major = prop.major,
      .minor = prop.minor,
    },
    thread_dimensions{
      .x = prop.maxThreadsDim[0],
      .y = prop.maxThreadsDim[1],
      .z = prop.maxThreadsDim[2],
    },
    grid_sizes{
      .x = prop.maxGridSize[0],
      .y = prop.maxGridSize[1],
      .z = prop.maxGridSize[2],
    },
    memory{
      .total_global = prop.totalGlobalMem,
      .total_const = prop.totalConstMem,
      .shared_per_block = prop.sharedMemPerBlock,
      .shared_per_multiprocessor = prop.sharedMemPerMultiprocessor,
    },
    registers{
      .per_block = prop.regsPerBlock,
      .per_multiprocessor = prop.regsPerMultiprocessor,
    }
  {
  }

  static DevicePropertiesWrapper wrap(DeviceProperties* ptr) {
    return { .ptr = ptr };
  }

  static DeviceProperties* unwrap(DevicePropertiesWrapper wrapper) {
    return static_cast<DeviceProperties*>(wrapper.ptr);
  }
  
  int multi_processor_count;
  int max_threads_per_multi_processor;
  int max_threads_per_block;
  PCIE pcie;
  ComputeVersion compute_version;
  ThreadDimensions thread_dimensions;
  GridSizes grid_sizes;
  DeviceMemory memory;
  DeviceRegisters registers;
};
  
//int deviceCount = 0;
//CUDA_ASSERT(cudaGetDeviceCount(&deviceCount));

//printf("Found %d CUDA-capable device(s).\n", deviceCount);

//for (int dev = 0; dev < deviceCount; ++dev) {
//    cudaDeviceProp deviceProp;
//    cudaGetDeviceProperties(&deviceProp, dev);

//    printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);
//    printf("  Compute capability:                %d.%d\n", deviceProp.major, deviceProp.minor);
//    printf("  PCI bus ID / device ID:            %d / %d\n", deviceProp.pciBusID, deviceProp.pciDeviceID);
//    printf("  Multi-Processor Count:             %d\n", deviceProp.multiProcessorCount);
//    printf("  Max Threads per Multi-Processor:   %d\n", deviceProp.maxThreadsPerMultiProcessor);
//    printf("  Max Threads per Block:             %d\n", deviceProp.maxThreadsPerBlock);
//    printf("  Max Block Dimensions:              %d x %d x %d\n",
//           deviceProp.maxThreadsDim[0],
//           deviceProp.maxThreadsDim[1],
//           deviceProp.maxThreadsDim[2]);
//    printf("  Max Grid Dimensions:               %d x %d x %d\n",
//           deviceProp.maxGridSize[0],
//           deviceProp.maxGridSize[1],
//           deviceProp.maxGridSize[2]);

//    printf("  Warp Size:                         %d\n", deviceProp.warpSize);
//    printf("  Clock Rate:                        %.2f MHz\n", deviceProp.clockRate / 1000.0f);
//    printf("  Memory Clock Rate:                 %.2f MHz\n", deviceProp.memoryClockRate / 1000.0f);
//    printf("  Memory Bus Width:                  %d bits\n", deviceProp.memoryBusWidth);
//    rintfPeak Memory Bandwidth (approx):    %.2f GB/s\n" {
//        .total_global = prop.totalGlobalMem,
// .total_const = prop.totalConstMem,      . 
// shared_per_block = prop.sharedMemPerBlock,
//     // },
  //   // 
//           2.0 * deviceProp.memoryClockRate * (deviceProp.memoryBusWidth / 8) / 1.0e6);

//    printf("  Total Global Memory:               %.2f GB\n", (float)deviceProp.totalGlobalMem / (1024 * 1024 * 1024));
//    printf("  Total Constant Memory:             %zu bytes\n", deviceProp.totalConstMem);
//    printf("  Shared Memory per Block:           %zu bytes\n", deviceProp.sharedMemPerBlock);
//    printf("  Shared Memory per SM:              %zu bytes\n", deviceProp.sharedMemPerMultiprocessor);

//    printf("  Registers per Block:               %d\n", deviceProp.regsPerBlock);
//    printf("  Registers per SM:                  %d\n", deviceProp.regsPerMultiprocessor);

//    //printf("  L2 Cache Size:                     %d bytes\n", deviceProp.l2CacheSize);
//    //printf("  Max Threads per SMP (Occupancy):   %d\n", deviceProp.maxThreadsPerMultiProcessor);

//    //// Whether the device can overlap kernel execution with data transfers
//    //printf("  Concurrent Kernels:                %s\n", deviceProp.concurrentKernels ? "Yes" : "No");
//    //printf("  Concurrent Copy and Kernel:        %s\n", deviceProp.deviceOverlap ? "Yes" : "No");

//    //printf("  ECC Enabled:                       %s\n", deviceProp.ECCEnabled ? "Yes" : "No");
//    //printf("  TCC Driver:                        %s\n", deviceProp.tccDriver ? "Yes" : "No");

//    // Compute Mode:
//    //  - cudaComputeModeDefault
//    //  - cudaComputeModeExclusive
//    //  - cudaComputeModeProhibited
//    //  - cudaComputeModeExclusiveProcess
//    //printf("  Compute Mode:                      ");
//    //switch (deviceProp.computeMode) {
//    //    case cudaComputeModeDefault:
//    //        printf("Default (multiple host threads can use this device simultaneously)\n");
//    //        break;
//    //    case cudaComputeModeExclusive:
//    //        printf("Exclusive (only one host thread at a time)\n");
//    //        break;
//    //    case cudaComputeModeProhibited:
//    //        printf("Prohibited (no host thread can use this device)\n");
//    //        break;
//    //    case cudaComputeModeExclusiveProcess:
//    //        printf("Exclusive Process (only one context used by a single process can access this device)\n");
//    //        break;
//    //    default:
//    //        printf("Unknown\n");
//    //        break;
//    //}

//    //printf("\n  ----- Recommendations for Kernel Launch -----\n");
//    //printf("  * Typical block size heuristics often start at multiples of %d (warp size)\n", deviceProp.warpSize);
//    //printf("  * For occupancy, consider registers/block, shared mem/block, and # SMs\n");
//    //printf("  * Occupancy-based tuning often yields block sizes of 128, 256, or 512\n");
//    //printf("  * Use cudaOccupancyMaxPotentialBlockSize or Nsight Compute for deeper analysis\n");
//    //printf("------------------------------------------------\n");
//
#endif
