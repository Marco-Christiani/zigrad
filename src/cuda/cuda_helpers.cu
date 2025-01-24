#ifndef __CUDA_HELPERS_ZIG__
#define __CUDA_HELPERS_ZIG__

#include <cutensor/types.h>
#include <stdio.h>
#include "/usr/local/cuda/include/cuda.h"
#include "/usr/local/cuda/include/cublas_v2.h"
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

inline CUstream get_stream(void* context) {
  return static_cast<CUstream>(context);
}

inline cublasHandle_t get_handle(void* context) {
  return static_cast<cublasHandle_t>(context);
}

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

template <typename T>
T* __alloc_scalar(cudaStream_t stream) {
  CUdeviceptr dptr;
  const CUstream _stream = static_cast<CUstream>(stream);
  CURESULT_ASSERT(cuMemAllocAsync(&dptr, sizeof(T), _stream));
  return reinterpret_cast<T*>(dptr);
}

template <typename T>
inline void __free_scalar(cudaStream_t stream, T* s) {
  CUstream _stream = static_cast<CUstream>(stream);
  CUdeviceptr dptr = reinterpret_cast<CUdeviceptr>(s);
  CURESULT_ASSERT(cuMemFreeAsync(dptr, _stream));
}

template <typename T>
T __transfer_scalar(cudaStream_t stream, T* s) {
  T result;
  CUstream _stream = static_cast<CUstream>(stream);
  CUdeviceptr dptr = reinterpret_cast<CUdeviceptr>(s);
  CURESULT_ASSERT(cuMemcpyDtoHAsync(&result, dptr, sizeof(T), _stream));
  CURESULT_ASSERT(cuStreamSynchronize(_stream));
  return result;
}

inline cudaStream_t __cublas_stream(void* handle) {
  cudaStream_t stream;
  CUBLAS_ASSERT(cublasGetStream(static_cast<cublasHandle_t>(handle), &stream));
  return stream;
}

inline cudaStream_t __cudnn_stream(void* handle) {
  cudaStream_t stream;
  CUDNN_ASSERT(cudnnGetStream(static_cast<cudnnHandle_t>(handle), &stream));
  return stream;
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

#endif

