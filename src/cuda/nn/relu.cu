#ifndef __NN_RELU_ZIG__
#define __NN_RELU_ZIG__

#include "nn_utils.cu"

extern "C" void relu_forward(
  dtype id,
  void* stream,
  const void* x,
  void* y,
  len_t n
) {
  const auto _stream = static_cast<cudaStream_t>(stream);

  switch (id) {
    case SINGLE: {
      const auto x_iter = static_cast<const float*>(x);
      const auto y_iter = static_cast<float*>(y);
      thrust::transform(
          thrust::cuda::par.on(_stream), 
          x_iter,  
          x_iter + n,
          y_iter,
          [] __device__ (float a) -> float { return (a > 0.0f) ? a : 0.0f; }
      );
      return;
    }
    case DOUBLE: {
      const auto x_iter = static_cast<const double*>(x);
      const auto y_iter = static_cast<double*>(y);
      thrust::transform(
          thrust::cuda::par.on(_stream), 
          x_iter,  
          x_iter + n,
          y_iter,
          [] __device__ (double a) -> float { return (a > 0.0) ? a : 0.0; }
      );
      return;
    }
  }
}

extern "C" void relu_reverse(
  dtype id,
  void* stream,
  const void* x,
  const void* y_grd,
  void* x_grd,
  len_t n
) {
  const auto _stream = static_cast<cudaStream_t>(stream);

  switch (id) {
    case SINGLE: {
      const auto x_iter = static_cast<const float*>(x);
      const auto y_grd_iter = static_cast<const float*>(y_grd);
      const auto x_grd_iter = static_cast<float*>(x_grd);
      thrust::transform(
          thrust::cuda::par.on(_stream), 
          x_iter,  
          x_iter + n,
          y_grd_iter,
          x_grd_iter,
          [] __device__ (float x, float yg) -> float { return (x > 0.0f) ? yg : 0.0f; }
      );
      return;
    }
    case DOUBLE: {
      const auto x_iter = static_cast<const double*>(x);
      const auto y_grd_iter = static_cast<const double*>(y_grd);
      const auto x_grd_iter = static_cast<double*>(x_grd);
      thrust::transform(
          thrust::cuda::par.on(_stream), 
          x_iter,  
          x_iter + n,
          y_grd_iter,
          x_grd_iter,
          [] __device__ (double x, double yg) -> double { return (x > 0.0f) ? yg : 0.0f; }
      );
      return;
    }
  }
}

#endif
