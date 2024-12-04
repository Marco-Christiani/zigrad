#ifndef __BLAS_MAX_ZIG__
#define __BLAS_MAX_ZIG__

#include "blas_utils.cu"

template <typename T>
__global__ void __max_fwd_set(const T* src, T* dst, const int* idx) {
  *dst = src[*idx];
}

template <typename T>
__global__ void __max_bwd_set(const T* y_grd, T* x_grd, const int* idx) {
  x_grd[*idx] = *y_grd;
}

extern "C" void max_forward(
  dtype id,
  void* handle,
  const void* x,
  void* y,
  int* idx,
  len_t n
) {
  const auto _handle = static_cast<cublasHandle_t>(handle);
  const auto _n = static_cast<int>(n);

  cudaStream_t stream = nullptr;
  CUBLAS_ASSERT(cublasGetStream(_handle, &stream));

  switch (id) {
    case SINGLE: {
      const auto _x = static_cast<const float*>(x);
      const auto _y = static_cast<float*>(y);
      CUBLAS_ASSERT(cublasIsamax(_handle, _n, _x, 1, idx));
      __max_fwd_set<float><<<1,1,0,stream>>>(_x, _y, idx);
    }
    case DOUBLE: {
      const auto _x = static_cast<const double*>(x);
      const auto _y = static_cast<double*>(y);
      CUBLAS_ASSERT(cublasIdamax(_handle, _n, _x, 1, idx));
      __max_fwd_set<double><<<1,1,0,stream>>>(_x, _y, idx);
    }
  }
}

extern "C" void max_reverse(
  dtype id,
  void* handle,
  const void* y_grd,
  void* x_grd,
  int* idx
) {
  const auto _handle = static_cast<cublasHandle_t>(handle);

  cudaStream_t stream = nullptr;
  CUBLAS_ASSERT(cublasGetStream(_handle, &stream));

  switch (id) {
    case SINGLE: {
      const auto _y_grd = static_cast<const float*>(y_grd);
      const auto _x_grd = static_cast<float*>(x_grd);
      __max_bwd_set<float><<<1,1,0,stream>>>(_y_grd, _x_grd, idx);
    }
    case DOUBLE: {
      const auto _y_grd = static_cast<const double*>(y_grd);
      const auto _x_grd = static_cast<double*>(x_grd);
      __max_bwd_set<double><<<1,1,0,stream>>>(_y_grd, _x_grd, idx);
    }
  }
}


#endif
