#ifndef __CUDA_HELPERS_ZIG__
#define __CUDA_HELPERS_ZIG__

#include <cutensor/types.h>
#include <stdio.h>
#include <cuda.h>
#include <cublas_v2.h>
#include "cuda_includes.cu"
#include "decls.h"

typedef unsigned char u8;
typedef float f32;
typedef double f64;
typedef int64_t i64;
typedef int32_t i32;
typedef uint64_t u64;
typedef uint32_t u32;

#define WARP_SIZE 32

#define CUDA_ASSERT(err) (HandleCudaError( err, __FILE__, __LINE__ ))
inline void HandleCudaError(cudaError_t err, const char *file, int line)
{
  if (err != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
    exit(EXIT_FAILURE);
  }
}

#define CUBLAS_ASSERT(err) (handleCublasError( err, __FILE__, __LINE__ ))
inline void handleCublasError(cublasStatus_t err, const char *file, int line)
{
  // TODO: Report better cublas errors
  if (err != CUBLAS_STATUS_SUCCESS) {
      printf("Cublas failure in %s at line %d\n", file, line);
    exit(EXIT_FAILURE);
  }
}

#define CURESULT_ASSERT(err) (handleCuresultError( err, __FILE__, __LINE__ ))
inline void handleCuresultError(CUresult err, const char *file, int line)
{
  if (err != CUDA_SUCCESS) {
    const char** msg = nullptr;

    cuGetErrorString(err, msg);

    if (*msg) {
      printf("%s in %s at line %d\n", *msg, file, line);
    } else {
      printf("Unkown error in %s at line %d\n", file, line);
    }   
    exit(EXIT_FAILURE);
  }
}

#define CUDNN_ASSERT(err) (handleCudnnError( err, __FILE__, __LINE__ ))
inline void handleCudnnError(cudnnStatus_t err, const char *file, int line)
{
  // TODO: Report better cublas errors
  if (err != CUDNN_STATUS_SUCCESS) {
      printf("CUDNN failure in %s at line %d\n", file, line);
    exit(EXIT_FAILURE);
  }
}

#define CUTENSOR_ASSERT(err) (handleCutensorStatus(err, __FILE__, __LINE__ ))
inline void handleCutensorStatus(cutensorStatus_t status, const char *file, int line)
{
  // TODO: Report better cublas errors
  if (status != CUTENSOR_STATUS_SUCCESS) {
      printf("%s in %s at line %d\n", cutensorGetErrorString(status), file, line);
    exit(EXIT_FAILURE);
  }
}

#define CHECK_INVARIANT(b, msg) (CheckInvariant(b, msg, __FILE__, __LINE__ ))
inline void CheckInvariant(bool check, const char* message, const char *file, int line)
{
  if (!(check)) {
    printf("%s in %s at line %d\n", (message), file, line);
    exit(EXIT_FAILURE);
  }
}

#define SYSTEM_EXIT(msg) (SystemExit(msg, __FILE__, __LINE__ ))
inline void SystemExit(const char* message, const char *file, int line)
{
  printf("%s in %s at line %d\n", (message), file, line);
  exit(EXIT_FAILURE);
}

template <typename T>
T* __alloc(cudaStream_t stream, len_t n, const char* file, int line) {
  CUdeviceptr dptr;
  const CUstream _stream = static_cast<CUstream>(stream);
  handleCuresultError(cuMemAllocAsync(&dptr, n * sizeof(T), _stream), file, line);
  return reinterpret_cast<T*>(dptr);
}

template <typename T>
inline void __free(cudaStream_t stream, T* s, const char* file, int line) {
  CUstream _stream = static_cast<CUstream>(stream);
  CUdeviceptr dptr = reinterpret_cast<CUdeviceptr>(s);
  handleCuresultError(cuMemFreeAsync(dptr, _stream), file, line);
}

template <typename T>
void __ensure_scratch(
  cudaStream_t stream,
  len_t* mem,
  len_t* cap,
  len_t new_cap,
  const char* file,
  int line
) {
  if (new_cap <= *cap)
    return;

  if (*cap > 0) {
    __free(stream, reinterpret_cast<T*>(*mem), file, line);
  }
  *mem = reinterpret_cast<len_t>(__alloc<T>(stream, new_cap, file, line));
  *cap = new_cap;
}

void __ensure_scratch(
  dtype id,
  cudaStream_t stream,
  len_t* mem,
  len_t* cap,
  len_t new_cap,
  const char* file,
  int line
) {
  switch (id) {
    case SINGLE: return __ensure_scratch<f32>(stream, mem, cap, new_cap, file, line);
    case DOUBLE: return __ensure_scratch<f64>(stream, mem, cap, new_cap, file, line);
    default: SYSTEM_EXIT("Unsupported datatype");
  }
}

#define ENSURE_SCRATCH(id, stream, mem, cap, new_cap)                    \
  (__ensure_scratch(id, stream, mem, cap, new_cap, __FILE__, __LINE__ )) \



inline CUstream __cast_custream(StreamWrapper w) {
  return static_cast<CUstream>(w.ptr);
}

inline cudnnHandle_t __cast_cudnn(CudnnWrapper w) {
  return static_cast<cudnnHandle_t>(w.ptr);
}

inline cublasHandle_t __cast_cublas(CublasWrapper w) {
  return static_cast<cublasHandle_t>(w.ptr);
}

inline cudaStream_t __cast_stream(StreamWrapper w) {
  return static_cast<cudaStream_t>(w.ptr);
}

inline cudaStream_t __cast_stream(CublasWrapper w) {
  cudaStream_t stream;
  CUBLAS_ASSERT(cublasGetStream(__cast_cublas(w), &stream));
  return stream;
}

inline cudaStream_t __cast_stream(CudnnWrapper w) {
  cudaStream_t stream;
  CUDNN_ASSERT(cudnnGetStream(__cast_cudnn(w), &stream));
  return stream;
}

inline len_t __product(const len_t* dims, len_t size) {
  len_t res = 1;
  for (len_t i = 0; i < size; ++i) {
    res *= dims[i];
  }
  return res;
}

#endif

