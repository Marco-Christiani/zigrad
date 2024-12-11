#include <stdio.h>
#include "cuda_utils.h"

#include "cuda_helpers.cu"

extern "C" void* memAlloc(len_t N, void* stream) {
  CUstream _stream = get_stream(stream);
  CUdeviceptr dptr;
  CURESULT_ASSERT(cuMemAllocAsync(&dptr, N, _stream));
  return (void*)dptr;
}

extern "C" void memcpyHtoD(void* dev_ptr, const void* cpu_ptr, len_t N, void* stream) {
  CUstream _stream = get_stream(stream);
  CUdeviceptr dptr = reinterpret_cast<CUdeviceptr>(dev_ptr);
  CURESULT_ASSERT(cuMemcpyHtoDAsync(dptr, cpu_ptr, N, _stream));
}

extern "C" void memcpyDtoH(void* cpu_ptr, void const* dev_ptr, len_t N, void* stream) {
  CUstream _stream = get_stream(stream);
  CUdeviceptr dptr = reinterpret_cast<CUdeviceptr>(dev_ptr);
  CURESULT_ASSERT(cuMemcpyDtoHAsync(cpu_ptr, dptr, N, _stream));
}

extern "C" void memcpyDtoD(void* dst_ptr, void const* src_ptr, len_t N, void* stream) {
  CUstream _stream = get_stream(stream);
  CUdeviceptr dst = reinterpret_cast<CUdeviceptr>(dst_ptr);
  CUdeviceptr src = reinterpret_cast<CUdeviceptr>(src_ptr);
  CURESULT_ASSERT(cuMemcpyDtoDAsync(dst, src, N, _stream));
}

extern "C" void memFree(void* dev_ptr, void* stream) {
  CUstream _stream = get_stream(stream);
  CUdeviceptr dptr = reinterpret_cast<CUdeviceptr>(dev_ptr);
  CURESULT_ASSERT(cuMemFreeAsync(dptr, _stream));
}

extern "C" void streamSynchronize(void* stream) {
  CUstream _stream = get_stream(stream);
  CURESULT_ASSERT(cuStreamSynchronize(_stream));
}

extern "C" void deviceSynchronize() {
  CUDA_ASSERT(cudaDeviceSynchronize());
}

extern "C" void initDevice(unsigned device_number) {

    CURESULT_ASSERT(cuInit(device_number));

    CUdevice device;
    CUcontext context;
    int device_count = 0;

    CURESULT_ASSERT(cuDeviceGetCount(&device_count));

    if (device_count <= device_number) {
        fprintf(stderr, "Error: no devices supporting CUDA\n");
        exit(-1);
    }

    CURESULT_ASSERT(cuDeviceGet(&device, device_number));
    CURESULT_ASSERT(cuCtxCreate(&context, 0, device));
}

// Convenience wrapper for cudaGetLastError.
// TODO: make this return values instead of void
extern "C" void checkLastError()
{
  CUDA_ASSERT(cudaDeviceSynchronize());
  auto err = cudaGetLastError();
  if (err != cudaSuccess) {
      fprintf(stderr, "Error %s: %s", cudaGetErrorName(err), cudaGetErrorString(err));
  }
}

extern "C" len_t deviceTotalMemory(unsigned device) {
  len_t total;
  CURESULT_ASSERT(cuDeviceTotalMem(&total, device));
  return total;
}

extern "C" void* initStream() {
  cudaStream_t cuda_stream = nullptr;

  // REMINDER: For multi-device support, we need to add a call to:
  //    CUresult cuCtxGetDevice ( CUdevice* device ) 

  CURESULT_ASSERT(cuStreamCreate(&cuda_stream, CU_STREAM_DEFAULT));
  return reinterpret_cast<void*>(cuda_stream);
}

extern "C" void deinitStream(void* stream) {
  // TODO: If devices get set, it's probably a good idea to capture
  //       which device a stream was created on and put that in the
  //       void* object. Research if it's required to deinit
  //       streams on the correct device.
  CURESULT_ASSERT(cuStreamDestroy(get_stream(stream)));
}

extern "C" void* initCublasHandle(void* stream) {
  cublasHandle_t blas_handle = nullptr;
  CUBLAS_ASSERT(cublasCreate(&blas_handle));
  CUBLAS_ASSERT(cublasSetStream(blas_handle, get_stream(stream)));
  return reinterpret_cast<void*>(blas_handle);
}

extern "C" void deinitCublasHandle(void* handle) {
  CUBLAS_ASSERT(cublasDestroy(get_handle(handle)));
}

extern "C" void* initCudnnHandle(void* stream) {
  cudnnHandle_t cudnn_handle = nullptr;
  CUDNN_ASSERT(cudnnCreate(&cudnn_handle));
  CUDNN_ASSERT(cudnnSetStream(cudnn_handle, get_stream(stream)));
  return reinterpret_cast<void*>(cudnn_handle);
}

extern "C" void deinitCudnnHandle(void* handle) {
  CUDNN_ASSERT(cudnnDestroy(static_cast<cudnnHandle_t>(handle)));
}

extern "C" void memFill(dtype id, void* data, len_t n, const void* value, void* stream) {
  
  const auto _stream = static_cast<cudaStream_t>(stream);

  switch (id) {
    case SINGLE: {
      const auto iter = static_cast<float*>(data);
      const auto _val = static_cast<const float*>(value);
      return thrust::fill(thrust::cuda::par.on(_stream), iter, iter + n, *_val);
    }
    case DOUBLE: {
      const auto iter = static_cast<double *>(data);
      const auto _val = static_cast<const double*>(value);
      return thrust::fill(thrust::cuda::par.on(_stream), iter, iter + n, *_val);
    }
  }
}

extern "C" void memSequence(dtype id, void* data, len_t n, const void* init, const void* step, void* stream) {
  
  const auto _stream = static_cast<cudaStream_t>(stream);

  switch (id) {
    case SINGLE: {
      const auto iter = static_cast<float*>(data);
      const auto _init = static_cast<const float*>(init);
      const auto _step = static_cast<const float*>(step);
      return thrust::sequence(thrust::cuda::par.on(_stream), iter, iter + n, *_init, *_step);
    }
    case DOUBLE: {
      const auto iter = static_cast<double*>(data);
      const auto _init = static_cast<const double*>(init);
      const auto _step = static_cast<const double*>(step);
      return thrust::sequence(thrust::cuda::par.on(_stream), iter, iter + n, *_init, *_step);
    }
  }
}

template <typename T>
struct UniformRandom {
    unsigned seed;
    __host__ __device__
    T operator()(unsigned n) const {
        thrust::default_random_engine rng(this->seed);
        thrust::uniform_real_distribution<T> dist(T(-1), T(1));
        rng.discard(n);
        return dist(rng);
    }
};

template <typename T>
struct NormalRandom {
    unsigned seed;
    __host__ __device__
    T operator()(unsigned n) const {
        thrust::default_random_engine rng(this->seed);
        thrust::normal_distribution<T> dist(T(0), T(1));
        rng.discard(n);
        return dist(rng);
    }
};

template <typename T>
void __mem_random(void* x, len_t n, randtype op, unsigned seed, void* stream) {
  const auto _stream = static_cast<cudaStream_t>(stream);
  thrust::counting_iterator<unsigned> idxs(0);
  if (op == UNIFORM) {
    thrust::transform(thrust::cuda::par.on(_stream), idxs, idxs+ n, static_cast<T*>(x), UniformRandom<T>{ .seed = seed });
  } else {
    thrust::transform(thrust::cuda::par.on(_stream), idxs, idxs+ n, static_cast<T*>(x), NormalRandom<T>{ .seed = seed });
  }
}

extern "C" void memRandom(dtype id, void* x, len_t n, randtype op, unsigned seed, void* stream) {
  if (id == SINGLE) {
    return __mem_random<float>(x, n, op, seed, stream);
  } else {
    return __mem_random<double>(x, n, op, seed, stream);
  }
}
