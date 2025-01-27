#include <stdio.h>
#include "cuda_utils.h"
#include "cuda_helpers.cu"
#include "device_properties.cu"

extern "C" void* mem_alloc(len_t N, void* stream) {
  CUstream _stream = get_stream(stream);
  CUdeviceptr dptr;
  CURESULT_ASSERT(cuMemAllocAsync(&dptr, N, _stream));
  return (void*)dptr;
}

extern "C" void memcpy_HtoD(void* dev_ptr, const void* cpu_ptr, len_t N, void* stream) {
  CUstream _stream = get_stream(stream);
  CUdeviceptr dptr = reinterpret_cast<CUdeviceptr>(dev_ptr);
  CURESULT_ASSERT(cuMemcpyHtoDAsync(dptr, cpu_ptr, N, _stream));
}

extern "C" void memcpy_DtoH(void* cpu_ptr, void const* dev_ptr, len_t N, void* stream) {
  CUstream _stream = get_stream(stream);
  CUdeviceptr dptr = reinterpret_cast<CUdeviceptr>(dev_ptr);
  CURESULT_ASSERT(cuMemcpyDtoHAsync(cpu_ptr, dptr, N, _stream));
}

extern "C" void memcpy_DtoD(void* dst_ptr, void const* src_ptr, len_t N, void* stream) {
  CUstream _stream = get_stream(stream);
  CUdeviceptr dst = reinterpret_cast<CUdeviceptr>(dst_ptr);
  CUdeviceptr src = reinterpret_cast<CUdeviceptr>(src_ptr);
  CURESULT_ASSERT(cuMemcpyDtoDAsync(dst, src, N, _stream));
}

extern "C" void mem_free(void* dev_ptr, void* stream) {
  CUstream _stream = get_stream(stream);
  CUdeviceptr dptr = reinterpret_cast<CUdeviceptr>(dev_ptr);
  CURESULT_ASSERT(cuMemFreeAsync(dptr, _stream));
}

extern "C" void stream_synchronize(void* stream) {
  CUstream _stream = get_stream(stream);
  CURESULT_ASSERT(cuStreamSynchronize(_stream));
}

extern "C" void device_synchronize() {
  CUDA_ASSERT(cudaDeviceSynchronize());
}

extern "C" DevicePropertiesWrapper init_device(unsigned device_number) {

    CURESULT_ASSERT(cuInit(device_number));

    CUdevice device;
    CUcontext context;
    int device_count = 0;

    CURESULT_ASSERT(cuDeviceGetCount(&device_count));

    CHECK_INVARIANT(device_count > 0, "Device count came back as zero");
    CHECK_INVARIANT(device_count > device_number, "Device ID greater than number of devices");

    CURESULT_ASSERT(cuDeviceGet(&device, device_number));
    CURESULT_ASSERT(cuCtxCreate(&context, 0, device));

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);

    return DeviceProperties::wrap(new DeviceProperties(deviceProp));
}

extern "C" void deinit_device(DevicePropertiesWrapper wrapper) {
    auto properties = DeviceProperties::unwrap(wrapper);
    delete properties;
}

// Convenience wrapper for cudaGetLastError.
// TODO: make this return values instead of void
extern "C" void check_last_error()
{
  CUDA_ASSERT(cudaDeviceSynchronize());
  auto err = cudaGetLastError();
  if (err != cudaSuccess) {
      fprintf(stderr, "Error %s: %s", cudaGetErrorName(err), cudaGetErrorString(err));
  }
}

extern "C" len_t device_total_memory(unsigned device) {
  len_t total;
  CURESULT_ASSERT(cuDeviceTotalMem(&total, device));
  return total;
}

extern "C" void* init_stream() {
  cudaStream_t cuda_stream = nullptr;

  // REMINDER: For multi-device support, we need to add a call to:
  //    CUresult cuCtxGetDevice ( CUdevice* device ) 

  CURESULT_ASSERT(cuStreamCreate(&cuda_stream, CU_STREAM_DEFAULT));
  return reinterpret_cast<void*>(cuda_stream);
}

extern "C" void deinit_stream(void* stream) {
  // TODO: If devices get set, it's probably a good idea to capture
  //       which device a stream was created on and put that in the
  //       void* object. Research if it's required to deinit
  //       streams on the correct device.
  CURESULT_ASSERT(cuStreamDestroy(get_stream(stream)));
}

extern "C" void* init_cublas_handle(void* stream) {
  cublasHandle_t blas_handle = nullptr;
  CUBLAS_ASSERT(cublasCreate(&blas_handle));
  CUBLAS_ASSERT(cublasSetStream(blas_handle, get_stream(stream)));
  return reinterpret_cast<void*>(blas_handle);
}

extern "C" void deinit_cublas_handle(void* handle) {
  CUBLAS_ASSERT(cublasDestroy(get_handle(handle)));
}

extern "C" void* init_cudnn_handle(void* stream) {
  cudnnHandle_t cudnn_handle = nullptr;
  CUDNN_ASSERT(cudnnCreate(&cudnn_handle));
  CUDNN_ASSERT(cudnnSetStream(cudnn_handle, get_stream(stream)));
  return reinterpret_cast<void*>(cudnn_handle);
}

extern "C" void deinit_cudnn_handle(void* handle) {
  CUDNN_ASSERT(cudnnDestroy(static_cast<cudnnHandle_t>(handle)));
}

#include "cutensor_utils.cu"

extern "C" CutensorWrapper init_cutensor_handle(void* stream) {
  CutensorBackend* backend = new CutensorBackend(stream);
  return CutensorBackend::wrap(backend);
}

extern "C" void deinit_cutensor_handle(CutensorWrapper wrapper) {
  CutensorBackend* backend = CutensorBackend::unwrap(wrapper);
  delete backend;
}

extern "C" void mem_fill(dtype id, void* data, len_t n, const void* value, void* stream) {
  
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

extern "C" void mem_sequence(dtype id, void* data, len_t n, const void* init, const void* step, void* stream) {
  
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

struct GraphBackend {
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t instance = nullptr;
};

extern "C" GraphWrapper open_capture(void* stream) {
    CUDA_ASSERT(cudaStreamBeginCapture(static_cast<cudaStream_t>(stream), cudaStreamCaptureModeGlobal));
    return GraphWrapper{ .ptr = new GraphBackend() };
}

extern "C" void save_capture(GraphWrapper wrapper, void* stream) {
    auto backend = static_cast<GraphBackend*>(wrapper.ptr);
    auto _stream = static_cast<cudaStream_t>(stream); 
    CUDA_ASSERT(cudaStreamEndCapture(_stream, &backend->graph));
    // TODO: Check out these "NULL, NULL, 0" arguments
    CUDA_ASSERT(cudaGraphInstantiate(&backend->instance, backend->graph, NULL, NULL, 0));
}

extern "C" void free_capture(GraphWrapper wrapper, void* stream) {
    auto backend = static_cast<GraphBackend*>(wrapper.ptr);
    CUDA_ASSERT(cudaGraphExecDestroy(backend->instance));
    CUDA_ASSERT(cudaGraphDestroy(backend->graph));
    delete backend;
}

extern "C" void run_capture(GraphWrapper wrapper, void* stream) {
    auto backend = static_cast<GraphBackend*>(wrapper.ptr);
    auto _stream = static_cast<cudaStream_t>(stream); 
    CUDA_ASSERT(cudaGraphLaunch(backend->instance, _stream));
}

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

extern "C" void mem_random(dtype id, void* x, len_t n, randtype op, unsigned seed, void* stream) {
  switch (id) {
    case SINGLE: return __mem_random<f32>(x, n, op, seed, stream);
    case DOUBLE: return __mem_random<f64>(x, n, op, seed, stream);
    default: SYSTEM_EXIT("Unsupported data type");
  }
}

template <typename T>
void __mem_take(
  const void* src,
  len_t src_len,
  const len_t* idxs,
  len_t idxs_len,
  void* dst,
  void* stream
){
  const auto _stream = static_cast<cudaStream_t>(stream);
  const auto src_itr = static_cast<const T*>(src);
  const auto dst_itr = static_cast<T*>(dst);
  thrust::counting_iterator<len_t> stencil(0);
  thrust::transform(
    thrust::cuda::par.on(_stream),
    stencil,
    stencil + idxs_len,
    dst_itr,
    [=]__device__(len_t i) -> T {
      return src_itr[idxs[i]];
    });
}

// assumes idx_len <= dst_len
extern "C" void mem_take(
  dtype id,
  const void* src,
  len_t src_len,
  const len_t* idxs,
  len_t idxs_len,
  void* dst,
  void* stream
) {  
  switch (id) {
    case SINGLE: return __mem_take<f32>(src, src_len, idxs, idxs_len, dst, stream);
    case DOUBLE: return __mem_take<f64>(src, src_len, idxs, idxs_len, dst, stream);
    default: SYSTEM_EXIT("Unsupported data type");
  }
}
