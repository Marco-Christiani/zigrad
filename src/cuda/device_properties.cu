#ifndef __ZIG_DEVICE_PROPERTIES__
#define __ZIG_DEVICE_PROPERTIES__

#include <stdio.h>
#include "cuda_helpers.cu"
#include <cuda_runtime.h>

typedef struct {
  len_t major;
  len_t minor;
} ComputeVersion;

typedef struct {
  len_t bus_id;
  len_t device_id;
} PCIE;

typedef struct {
  size_t total_global;
  size_t total_const;
  size_t shared_per_block;
  size_t shared_per_multiprocessor;
} DeviceMemory;

typedef struct {
  len_t per_block;
  len_t per_multiprocessor;
} DeviceRegisters;

typedef struct {
  len_t x;
  len_t y;
  len_t z;
} GridSizes;

typedef struct {
  len_t x;
  len_t y;
  len_t z;
} ThreadDimensions;

// it's extremely annoying to cast from ints to calculate
// unsigned sizes. I'm casting them here to get it over with.

struct DeviceProperties {

  DeviceProperties(cudaDeviceProp prop) :
    multi_processor_count{ (len_t)prop.multiProcessorCount },
    max_threads_per_multi_processor{ (len_t)prop.maxThreadsPerMultiProcessor },
    max_threads_per_block{ (len_t)prop.maxThreadsPerBlock },
    pcie{
      .bus_id = (len_t)prop.pciBusID,
      .device_id = (len_t)prop.pciDeviceID,
    },
    compute_version{
      .major = (len_t)prop.major,
      .minor = (len_t)prop.minor,
    },
    thread_dimensions{
      .x = (len_t)prop.maxThreadsDim[0],
      .y = (len_t)prop.maxThreadsDim[1],
      .z = (len_t)prop.maxThreadsDim[2],
    },
    grid_sizes{
      .x = (len_t)prop.maxGridSize[0],
      .y = (len_t)prop.maxGridSize[1],
      .z = (len_t)prop.maxGridSize[2],
    },
    memory{
      .total_global = prop.totalGlobalMem,
      .total_const = prop.totalConstMem,
      .shared_per_block = prop.sharedMemPerBlock,
      .shared_per_multiprocessor = prop.sharedMemPerMultiprocessor,
    },
    registers{
      .per_block = (len_t)prop.regsPerBlock,
      .per_multiprocessor = (len_t)prop.regsPerMultiprocessor,
    }
  {
  }

  static DevicePropertiesWrapper wrap(DeviceProperties* ptr) {
    return { .ptr = ptr };
  }

  static DeviceProperties* unwrap(DevicePropertiesWrapper wrapper) {
    return static_cast<DeviceProperties*>(wrapper.ptr);
  }
  
  len_t multi_processor_count;
  len_t max_threads_per_multi_processor;
  len_t max_threads_per_block;
  PCIE pcie;
  ComputeVersion compute_version;
  ThreadDimensions thread_dimensions;
  GridSizes grid_sizes;
  DeviceMemory memory;
  DeviceRegisters registers;
};
  
//len_t deviceCount = 0;
//CUDA_ASSERT(cudaGetDeviceCount(&deviceCount));

//prlen_tf("Found %d CUDA-capable device(s).\n", deviceCount);

//for (len_t dev = 0; dev < deviceCount; ++dev) {
//    cudaDeviceProp deviceProp;
//    cudaGetDeviceProperties(&deviceProp, dev);

//    prlen_tf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);
//    prlen_tf("  Compute capability:                %d.%d\n", deviceProp.major, deviceProp.minor);
//    prlen_tf("  PCI bus ID / device ID:            %d / %d\n", deviceProp.pciBusID, deviceProp.pciDeviceID);
//    prlen_tf("  Multi-Processor Count:             %d\n", deviceProp.multiProcessorCount);
//    prlen_tf("  Max Threads per Multi-Processor:   %d\n", deviceProp.maxThreadsPerMultiProcessor);
//    prlen_tf("  Max Threads per Block:             %d\n", deviceProp.maxThreadsPerBlock);
//    prlen_tf("  Max Block Dimensions:              %d x %d x %d\n",
//           deviceProp.maxThreadsDim[0],
//           deviceProp.maxThreadsDim[1],
//           deviceProp.maxThreadsDim[2]);
//    prlen_tf("  Max Grid Dimensions:               %d x %d x %d\n",
//           deviceProp.maxGridSize[0],
//           deviceProp.maxGridSize[1],
//           deviceProp.maxGridSize[2]);

//    prlen_tf("  Warp Size:                         %d\n", deviceProp.warpSize);
//    prlen_tf("  Clock Rate:                        %.2f MHz\n", deviceProp.clockRate / 1000.0f);
//    prlen_tf("  Memory Clock Rate:                 %.2f MHz\n", deviceProp.memoryClockRate / 1000.0f);
//    prlen_tf("  Memory Bus Width:                  %d bits\n", deviceProp.memoryBusWidth);
//    rlen_tfPeak Memory Bandwidth (approx):    %.2f GB/s\n" {
//        .total_global = prop.totalGlobalMem,
// .total_const = prop.totalConstMem,      . 
// shared_per_block = prop.sharedMemPerBlock,
//     // },
  //   // 
//           2.0 * deviceProp.memoryClockRate * (deviceProp.memoryBusWidth / 8) / 1.0e6);

//    prlen_tf("  Total Global Memory:               %.2f GB\n", (float)deviceProp.totalGlobalMem / (1024 * 1024 * 1024));
//    prlen_tf("  Total Constant Memory:             %zu bytes\n", deviceProp.totalConstMem);
//    prlen_tf("  Shared Memory per Block:           %zu bytes\n", deviceProp.sharedMemPerBlock);
//    prlen_tf("  Shared Memory per SM:              %zu bytes\n", deviceProp.sharedMemPerMultiprocessor);

//    prlen_tf("  Registers per Block:               %d\n", deviceProp.regsPerBlock);
//    prlen_tf("  Registers per SM:                  %d\n", deviceProp.regsPerMultiprocessor);

//    //prlen_tf("  L2 Cache Size:                     %d bytes\n", deviceProp.l2CacheSize);
//    //prlen_tf("  Max Threads per SMP (Occupancy):   %d\n", deviceProp.maxThreadsPerMultiProcessor);

//    //// Whether the device can overlap kernel execution with data transfers
//    //prlen_tf("  Concurrent Kernels:                %s\n", deviceProp.concurrentKernels ? "Yes" : "No");
//    //prlen_tf("  Concurrent Copy and Kernel:        %s\n", deviceProp.deviceOverlap ? "Yes" : "No");

//    //prlen_tf("  ECC Enabled:                       %s\n", deviceProp.ECCEnabled ? "Yes" : "No");
//    //prlen_tf("  TCC Driver:                        %s\n", deviceProp.tccDriver ? "Yes" : "No");

//    // Compute Mode:
//    //  - cudaComputeModeDefault
//    //  - cudaComputeModeExclusive
//    //  - cudaComputeModeProhibited
//    //  - cudaComputeModeExclusiveProcess
//    //prlen_tf("  Compute Mode:                      ");
//    //switch (deviceProp.computeMode) {
//    //    case cudaComputeModeDefault:
//    //        prlen_tf("Default (multiple host threads can use this device simultaneously)\n");
//    //        break;
//    //    case cudaComputeModeExclusive:
//    //        prlen_tf("Exclusive (only one host thread at a time)\n");
//    //        break;
//    //    case cudaComputeModeProhibited:
//    //        prlen_tf("Prohibited (no host thread can use this device)\n");
//    //        break;
//    //    case cudaComputeModeExclusiveProcess:
//    //        prlen_tf("Exclusive Process (only one context used by a single process can access this device)\n");
//    //        break;
//    //    default:
//    //        prlen_tf("Unknown\n");
//    //        break;
//    //}

//    //prlen_tf("\n  ----- Recommendations for Kernel Launch -----\n");
//    //prlen_tf("  * Typical block size heuristics often start at multiples of %d (warp size)\n", deviceProp.warpSize);
//    //prlen_tf("  * For occupancy, consider registers/block, shared mem/block, and # SMs\n");
//    //prlen_tf("  * Occupancy-based tuning often yields block sizes of 128, 256, or 512\n");
//    //prlen_tf("  * Use cudaOccupancyMaxPotentialBlockSize or Nsight Compute for deeper analysis\n");
//    //prlen_tf("------------------------------------------------\n");
//
#endif
