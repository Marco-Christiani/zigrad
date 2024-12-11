#ifndef __CUDA_HELPERS_ZIG__
#define __CUDA_HELPERS_ZIG__

#include <stdio.h>
#include "/usr/local/cuda/include/cuda.h"
#include "/usr/local/cuda/include/cublas_v2.h"
#include "cuda_includes.cu"
#include "decls.h"

typedef float f32;
typedef double f64;

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
