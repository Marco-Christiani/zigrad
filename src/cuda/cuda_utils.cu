#include <stdio.h>
#include "cuda_utils.h"
#include "cuda_helpers.cu"
#include "decls.h"
#include "device_properties.cu"
#include <vector>
#include <thread>

extern "C" void* mem_alloc(len_t N, StreamWrapper stream) {
  CUstream _stream = __cast_stream(stream);
  CUdeviceptr dptr;
  CURESULT_ASSERT(cuMemAllocAsync(&dptr, N, _stream));
  return (void*)dptr;
}

extern "C" void memcpy_HtoD(void* dev_ptr, const void* cpu_ptr, len_t N, StreamWrapper stream) {
  CUstream _stream = __cast_stream(stream);
  CUdeviceptr dptr = reinterpret_cast<CUdeviceptr>(dev_ptr);
  CURESULT_ASSERT(cuMemcpyHtoDAsync(dptr, cpu_ptr, N, _stream));
}

extern "C" void memcpy_DtoH(void* cpu_ptr, void const* dev_ptr, len_t N, StreamWrapper stream) {
  CUstream _stream = __cast_stream(stream);
  CUdeviceptr dptr = reinterpret_cast<CUdeviceptr>(dev_ptr);
  CURESULT_ASSERT(cuMemcpyDtoHAsync(cpu_ptr, dptr, N, _stream));
}

extern "C" void memcpy_DtoD(void* dst_ptr, void const* src_ptr, len_t N, StreamWrapper stream) {
  CUstream _stream = __cast_stream(stream);
  CUdeviceptr dst = reinterpret_cast<CUdeviceptr>(dst_ptr);
  CUdeviceptr src = reinterpret_cast<CUdeviceptr>(src_ptr);
  CURESULT_ASSERT(cuMemcpyDtoDAsync(dst, src, N, _stream));
}

extern "C" void mem_free(void* dev_ptr, StreamWrapper stream) {
  CUstream _stream = __cast_stream(stream);
  CUdeviceptr dptr = reinterpret_cast<CUdeviceptr>(dev_ptr);
  CURESULT_ASSERT(cuMemFreeAsync(dptr, _stream));
}

extern "C" void stream_synchronize(StreamWrapper stream) {
  CUstream _stream = __cast_stream(stream);
  CURESULT_ASSERT(cuStreamSynchronize(_stream));
}

extern "C" void device_synchronize() {
  CUDA_ASSERT(cudaDeviceSynchronize());
}

extern "C" unsigned device_count() {
    CURESULT_ASSERT(cuInit(0));
    int device_count;
    CURESULT_ASSERT(cuDeviceGetCount(&device_count));
    return static_cast<unsigned>(device_count);
}

extern "C" DevicePropertiesWrapper init_device(unsigned device_number) {

    CUdevice device;
    CUcontext context;
    int device_count = 0;

    CURESULT_ASSERT(cuInit(0));
    CURESULT_ASSERT(cuDeviceGetCount(&device_count));

    CHECK_INVARIANT(device_count > 0, "Device count came back as zero");
    CHECK_INVARIANT(device_count > device_number, "Device ID greater than number of devices");

    CURESULT_ASSERT(cuDeviceGet(&device, device_number));
    CURESULT_ASSERT(cuCtxCreate(&context, 0, device));

    cudaDeviceProp deviceProp;
    CUDA_ASSERT(cudaGetDeviceProperties(&deviceProp, device));

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

extern "C" StreamWrapper init_stream() {
  cudaStream_t cuda_stream = nullptr;

  // REMINDER: For multi-device support, we need to add a call to:
  //    CUresult cuCtxGetDevice ( CUdevice* device ) 

  CURESULT_ASSERT(cuStreamCreate(&cuda_stream, CU_STREAM_DEFAULT));
  return { .ptr = cuda_stream };
}

extern "C" void deinit_stream(StreamWrapper stream) {
  // TODO: If devices get set, it's probably a good idea to capture
  //       which device a stream was created on and put that in the
  //       void* object. Research if it's required to deinit
  //       streams on the correct device.
  CURESULT_ASSERT(cuStreamDestroy(__cast_stream(stream)));
}

extern "C" CublasWrapper init_cublas_handle(StreamWrapper stream) {
  cublasHandle_t blas_handle = nullptr;
  CUBLAS_ASSERT(cublasCreate(&blas_handle));
  CUBLAS_ASSERT(cublasSetStream(blas_handle, __cast_stream(stream)));
  return { .ptr = blas_handle };
}

extern "C" void deinit_cublas_handle(CublasWrapper w) {
  CUBLAS_ASSERT(cublasDestroy(__cast_cublas(w)));
}

extern "C" CudnnWrapper init_cudnn_handle(StreamWrapper stream) {
  cudnnHandle_t cudnn_handle = nullptr;
  CUDNN_ASSERT(cudnnCreate(&cudnn_handle));
  CUDNN_ASSERT(cudnnSetStream(cudnn_handle, __cast_stream(stream)));
  return { .ptr = cudnn_handle };
}

extern "C" void deinit_cudnn_handle(CudnnWrapper w) {
  CUDNN_ASSERT(cudnnDestroy(__cast_cudnn(w)));
}

#include "cutensor_utils.cu"

extern "C" CutensorWrapper init_cutensor_handle(StreamWrapper stream) {
  CutensorBackend* backend = new CutensorBackend(__cast_stream(stream));
  return CutensorBackend::wrap(backend);
}

extern "C" void deinit_cutensor_handle(CutensorWrapper wrapper) {
  CutensorBackend* backend = CutensorBackend::unwrap(wrapper);
  delete backend;
}

extern "C" void mem_fill(dtype id, void* data, len_t n, const void* value, StreamWrapper stream) {
  
  const auto _stream = __cast_stream(stream);

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

extern "C" void mem_sequence(dtype id, void* data, len_t n, const void* init, const void* step, StreamWrapper stream) {
  
  const auto _stream = __cast_stream(stream);

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

extern "C" GraphWrapper open_capture(StreamWrapper stream) {
    CUDA_ASSERT(cudaStreamBeginCapture(__cast_stream(stream), cudaStreamCaptureModeGlobal));
    return GraphWrapper{ .ptr = new GraphBackend() };
}

extern "C" void save_capture(GraphWrapper wrapper, StreamWrapper stream) {
    auto _stream = __cast_stream(stream); 
    auto backend = static_cast<GraphBackend*>(wrapper.ptr);
    CUDA_ASSERT(cudaStreamEndCapture(_stream, &backend->graph));
    // TODO: Check out these "NULL, NULL, 0" arguments
    CUDA_ASSERT(cudaGraphInstantiate(&backend->instance, backend->graph, NULL, NULL, 0));
}

extern "C" void free_capture(GraphWrapper wrapper, StreamWrapper stream) {
    auto backend = static_cast<GraphBackend*>(wrapper.ptr);
    CUDA_ASSERT(cudaGraphExecDestroy(backend->instance));
    CUDA_ASSERT(cudaGraphDestroy(backend->graph));
    delete backend;
}

extern "C" void run_capture(GraphWrapper wrapper, StreamWrapper stream) {
    auto _stream = __cast_stream(stream);
    auto backend = static_cast<GraphBackend*>(wrapper.ptr);
    CUDA_ASSERT(cudaGraphLaunch(backend->instance, _stream));
}

template <typename T>
void __mem_random(void* x, len_t n, randtype op, unsigned seed, StreamWrapper stream) {
  const auto _stream = __cast_stream(stream);
  thrust::counting_iterator<unsigned> idxs(0);
  if (op == UNIFORM) {
    thrust::transform(thrust::cuda::par.on(_stream), idxs, idxs+ n, static_cast<T*>(x), UniformRandom<T>{ .seed = seed });
  } else {
    thrust::transform(thrust::cuda::par.on(_stream), idxs, idxs+ n, static_cast<T*>(x), NormalRandom<T>{ .seed = seed });
  }
}

extern "C" void mem_random(dtype id, void* x, len_t n, randtype op, unsigned seed, StreamWrapper stream) {
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
  StreamWrapper stream
){
  const auto _stream = __cast_stream(stream);
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
  StreamWrapper stream
) {  
  switch (id) {
    case SINGLE: return __mem_take<f32>(src, src_len, idxs, idxs_len, dst, stream);
    case DOUBLE: return __mem_take<f64>(src, src_len, idxs, idxs_len, dst, stream);
    default: SYSTEM_EXIT("Unsupported data type");
  }
}


// Helper to keep track of cuda mem-handles for mapping
// physical regions to virtual device addresses. This
// currently grows monotonically overtime. We can support
// unmapping segments in a future release.
struct MapEntry {
  CUdeviceptr dptr;
  len_t size;
  CUmemGenericAllocationHandle handle;
};

// TODO: At some point, it makes sense to allow access to
// the entries in this struct directly.
struct Memmap {  
  CUmemAllocationProp prop;
  CUmemAccessDesc access_desc;
  std::vector<MapEntry> handles;

  static Memmap* unwrap(MemmapWrapper wrapper) {
    return static_cast<Memmap*>(wrapper.ptr);
  }

  static MemmapWrapper wrap(Memmap* backend) {
    return { .ptr = backend };
  }

  Memmap(unsigned device_id) {
      this->prop = {};
      this->prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
      this->prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
      this->prop.location.id = static_cast<int>(device_id);

      this->access_desc = {};
      this->access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
      this->access_desc.location.id = static_cast<int>(device_id);
      this->access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

      this->handles = {};
  }

  void map_alloc_impl(void* address, len_t size) {

    CUdeviceptr dptr = reinterpret_cast<CUdeviceptr>(address);

    CUmemGenericAllocationHandle handle;
    CURESULT_ASSERT(cuMemCreate(&handle, size, &this->prop, 0));
    CURESULT_ASSERT(cuMemMap(dptr, size, 0, handle, 0));
    CURESULT_ASSERT(cuMemSetAccess(dptr, size, &this->access_desc, 1));

    this->handles.emplace_back(MapEntry{ .dptr = dptr, .size = size, .handle = handle });
  }

  void reset() {
    for (auto& entry: this->handles) {
      CURESULT_ASSERT(cuMemUnmap(entry.dptr, entry.size));
      CURESULT_ASSERT(cuMemRelease(entry.handle));
    }
    this->handles.clear();
  }

  ~Memmap() {
    this->reset();
  }
};

extern "C" MemmapWrapper init_memmap(unsigned device_id) {
  return Memmap::wrap(new Memmap(device_id));
}

extern "C" void reset_memmap(MemmapWrapper wrapper) {
  Memmap::unwrap(wrapper)->reset();
}

extern "C" void deinit_memmap(MemmapWrapper wrapper) {
  delete Memmap::unwrap(wrapper);
}

extern "C" void mem_map_alloc(MemmapWrapper wrapper, void* address, len_t size) {
  Memmap::unwrap(wrapper)->map_alloc_impl(address, size);
}

extern "C" len_t mem_page_size(unsigned device_id) {
  CUmemAllocationProp prop;
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = static_cast<int>(device_id);

  len_t page_size;
  CURESULT_ASSERT(cuMemGetAllocationGranularity(&page_size, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  return page_size;
}


extern "C" void* mem_map(unsigned device_id, size_t virtual_buffer_size) {
  CUdeviceptr base_address = 0;  
  // This is an annoying hack because "cuMemAddressReserve" only works with the
  // device assigned to the current thread context. Directly changing will tamper
  // with objects observing the previous context. Create a child thread, set its
  // context, and then reserve space on the specified device.
 std::thread ctx_thread([=](CUdeviceptr* dptr_ref){
      CUcontext ctx;
      CURESULT_ASSERT(cuDevicePrimaryCtxRetain(&ctx, static_cast<int>(device_id)));
      CURESULT_ASSERT(cuCtxSetCurrent(ctx));
      CURESULT_ASSERT(cuMemAddressReserve(dptr_ref, virtual_buffer_size, mem_page_size(device_id), 0, 0));

      //CUmemAllocationProp prop;
      //CUmemAccessDesc access_desc;

      //prop = {};
      //prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
      //prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
      //prop.location.id = static_cast<int>(device_id);

      //access_desc = {};
      //access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
      //access_desc.location.id = static_cast<int>(device_id);
      //access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

      //

      //CUmemGenericAllocationHandle handle;
      //CURESULT_ASSERT(cuMemCreate(&handle, mem_page_size(device_id), &prop, 0));
      //CURESULT_ASSERT(cuMemMap(*dptr_ref, mem_page_size(device_id), 0, handle, 0));
      //CURESULT_ASSERT(cuMemSetAccess(*dptr_ref, mem_page_size(device_id), &access_desc, 1));

 }, &base_address);

  ctx_thread.join();
  
  return reinterpret_cast<void*>(base_address);
}


extern "C" void mem_unmap(unsigned device_id, void* base_address, size_t virtual_buffer_size) {
  // This is an annoying hack because "cuMemAddressReserve" only works with the
  // device assigned to the current thread context. Directly changing will tamper
  // with objects observing the previous context. Create a child thread, set its
  // context, and then free the space associated with this device.
  std::thread ctx_thread([=](){
      CUcontext ctx;
      CURESULT_ASSERT(cuDevicePrimaryCtxRetain(&ctx, static_cast<int>(device_id)));
      CURESULT_ASSERT(cuCtxSetCurrent(ctx));
      CUdeviceptr dptr = reinterpret_cast<CUdeviceptr>(base_address);
      CURESULT_ASSERT(cuMemAddressFree(dptr, virtual_buffer_size));
  });

  ctx_thread.join();
}













