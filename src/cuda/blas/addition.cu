#ifndef __BLAS_ADDITION_ZIG__
#define __BLAS_ADDITION_ZIG__

#include "blas_utils.cu"

// we're using double because every float can cast
// up to a double and then we can go back down.

template<typename T>
struct WrappingAdd {

    const T* x;
    const T* y;
    len_t x_len;
    len_t y_len;

    __device__
    T operator()(len_t i) const {
      return this->x[i % this->x_len] + this->y[i % this->y_len];
    }
};


template <class T>
void __addition(
  StreamWrapper w,
  const void* x,
  const void* y,
  void* z,
  len_t x_len,
  len_t y_len,
  len_t z_len
) {  
  const auto _stream = __cast_stream(w);
  const auto counter = thrust::make_counting_iterator<len_t>(0ul);
  const auto iter_z = static_cast<T*>(z);
  thrust::transform(
    thrust::cuda::par.on(_stream), 
    counter,  
    counter + z_len,
    iter_z,
    WrappingAdd<T>{
      .x = static_cast<const T*>(x),
      .y = static_cast<const T*>(y),
      .x_len = x_len,
      .y_len = y_len
    }
  );
}

extern "C" void addition(
  dtype id,
  StreamWrapper w,
  const void* x,
  const void* y,
  void* z,
  len_t x_len,
  len_t y_len,
  len_t z_len
) {

  switch (id) {
    case SINGLE: {
        return __addition<f32>(w, x, y, z, x_len, y_len, z_len);
    }
    case DOUBLE: {
        return __addition<f64>(w, x, y, z, x_len, y_len, z_len);
    }
  }
  CUDA_ASSERT(cudaPeekAtLastError());
}

#endif
