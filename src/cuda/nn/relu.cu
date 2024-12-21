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

template <typename T>
struct ReluBwdFunctor {
    __device__
    T operator()(thrust::tuple<T, T> tup, T x_grd) const {
        return ((thrust::get<0>(tup) > T{0}) ? thrust::get<1>(tup) : T{0}) + x_grd;
    }
};

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
      const auto z_iter = thrust::make_zip_iterator(thrust::make_tuple(x_iter, y_grd_iter));
      thrust::transform(
          thrust::cuda::par.on(_stream), 
          z_iter,
          z_iter + n,
          x_grd_iter,
          x_grd_iter,
          ReluBwdFunctor<float>{}
      );
      return;
    }
    case DOUBLE: {
      const auto x_iter = static_cast<const double*>(x);
      const auto y_grd_iter = static_cast<const double*>(y_grd);
      const auto x_grd_iter = static_cast<double*>(x_grd);
      const auto z_iter = thrust::make_zip_iterator(thrust::make_tuple(x_iter, y_grd_iter));
      thrust::transform(
          thrust::cuda::par.on(_stream), 
          z_iter,
          z_iter + n,
          x_grd_iter,
          x_grd_iter,
          ReluBwdFunctor<double>{}
      );
      return;
    }
  }
}

#endif
