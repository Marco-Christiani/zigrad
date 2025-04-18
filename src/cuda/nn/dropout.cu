#ifndef __NN_DROPOUT_ZIG__
#define __NN_DROPOUT_ZIG__

#include "nn_utils.cu"
#include "curand_kernel.h"

__device__
uint8_t __set_bit(uint8_t pos, bool value) {
  return (static_cast<uint8_t>(value) << pos);
}

template <typename T> __device__
T __check_bit(uint8_t mask, uint8_t pos) {
  return static_cast<T>(((mask & (1 << pos)) != 0));
}

// try to reduce register space - only create one float4 at a time
__device__ uint8_t __rand_bits(curandStatePhilox4_32_10_t* state, float p) {
  uint8_t bits = 0;
  float4 r4 = curand_uniform4(state);
  bits |= __set_bit(0, r4.w < p);
  bits |= __set_bit(1, r4.x < p);
  bits |= __set_bit(2, r4.y < p);
  bits |= __set_bit(3, r4.z < p);
  r4 = curand_uniform4(state);
  bits |= __set_bit(4, r4.w < p);
  bits |= __set_bit(5, r4.x < p);
  bits |= __set_bit(6, r4.y < p);
  bits |= __set_bit(7, r4.z < p);
  return bits;
}

__device__
uint64_t __mask_index(uint64_t index) {
  return index / 8;
}

__global__ void __populate_dropout(
  uint64_t* __restrict int_mask,
  len_t int_mask_len,
  float p,
  uint64_t seed
) {
  uint8_t* byte_mask = reinterpret_cast<uint8_t*>(int_mask);
  len_t byte_mask_len = int_mask_len * 8;

  const len_t g_len = gridDim.x * blockDim.x;
  len_t t_pos = blockDim.x * blockIdx.x + threadIdx.x;
  
  curandStatePhilox4_32_10_t state;
  curand_init(seed, t_pos, 0, &state);

  for (; t_pos < byte_mask_len; t_pos += g_len) {
    byte_mask[t_pos] = __rand_bits(&state, p);
  }
}

template <class T>
__global__ void __dropout_1d_forward_kernel(
  const T* __restrict x,
        T* __restrict y,
  uint64_t x_len,
  uint64_t* __restrict int_mask,
  float p
) {
  const auto byte_mask = reinterpret_cast<uint8_t* __restrict__>(int_mask);
  const float scale = 1.0 / p;

  const uint64_t g_len = gridDim.x * blockDim.x;

  // each thread handles 1-byte (8 elements of randomness)
  uint64_t t_pos = blockDim.x * blockIdx.x + threadIdx.x;

  // TODO: This definitely needs to be vectorized.
  for (; t_pos < x_len; t_pos += g_len) {
    const uint8_t byte = byte_mask[__mask_index(t_pos)];
    y[t_pos] = x[t_pos] * __check_bit<T>(byte, t_pos) * scale;
  }
}


template <class T>
__global__ void __dropout_1d_backward_kernel(
  const T* __restrict y_grad,
        T* __restrict x_grad,
  uint64_t x_len,
  uint64_t const* __restrict int_mask,
  float p
) {
    const auto byte_mask = reinterpret_cast<const uint8_t* __restrict__>(int_mask);
    const float scale = 1.0 / p;

    const uint64_t g_len = gridDim.x * blockDim.x;

    // each thread handles 1-byte (8 elements of randomness)
    uint64_t t_pos = blockDim.x * blockIdx.x + threadIdx.x;

    // TODO: This definitely needs to be vectorized.
    for (; t_pos < x_len; t_pos += g_len) {
      const uint8_t byte = byte_mask[__mask_index(t_pos)];
      x_grad[t_pos] = y_grad[t_pos] * __check_bit<T>(byte, t_pos) * scale;
    }
}

template<typename T>
inline T __ceil_div(T x, T y) {
  return ((x - 1) / y) + 1;
}

template <class T>
 void __dropout_1d_forward(
  cudaStream_t stream,
  const DeviceProperties* props,
  const void* x,
        void* y,
  uint64_t x_len,
  uint64_t* int_mask,
  float p
) {
  const len_t threads = std::min(x_len, props->max_threads_per_block);
  const len_t blocks = std::min(__ceil_div(x_len, threads), props->grid_sizes.x);    
  __dropout_1d_forward_kernel<T><<<threads,blocks,0,stream>>>(static_cast<const T*>(x), static_cast<T*>(y), x_len, int_mask, p);
  CUDA_ASSERT(cudaPeekAtLastError());
}

extern "C" void dropout_1d_forward(
  dtype id,
  void* _stream,
  DevicePropertiesWrapper _props,
  const void* x,
  void* y,
  len_t x_len,
  uint64_t* int_mask,
  len_t int_mask_len,
  float p,
  uint64_t seed
) {
  const auto stream = static_cast<cudaStream_t>(_stream);
  const auto props = DeviceProperties::unwrap(_props);

  { // This may be faster if we combine populating the dropout with applying it.
    const len_t byte_mask_len = int_mask_len * 8;
    const len_t threads = std::min(byte_mask_len, props->max_threads_per_block);
    const len_t blocks = std::min(__ceil_div(byte_mask_len, threads), props->grid_sizes.x);    
    __populate_dropout<<<threads,blocks,0,stream>>>(int_mask, int_mask_len, p, seed);
    CUDA_ASSERT(cudaPeekAtLastError());
  }

  switch (id) {
    case SINGLE: {
        return __dropout_1d_forward<f32>(stream, props, x, y, x_len, int_mask, p);
    }
    case DOUBLE: {
        return __dropout_1d_forward<f64>(stream, props, x, y, x_len, int_mask, p);
    }
    default:
      SYSTEM_EXIT("Unsupported Data Type");
  }
}

template <class T>
 void __dropout_1d_backward(
  cudaStream_t stream,
  const DeviceProperties* props,
  const void* y_grad,
        void* x_grad,
  uint64_t x_len,
  uint64_t* int_mask,
  float p
) {
  const len_t threads = std::min(x_len, props->max_threads_per_block);
  const len_t blocks = std::min(__ceil_div(x_len, threads), props->grid_sizes.x);    
  __dropout_1d_backward_kernel<T><<<threads,blocks,0,stream>>>(static_cast<const T*>(y_grad), static_cast<T*>(x_grad), x_len, int_mask, p);
  CUDA_ASSERT(cudaPeekAtLastError());
}

extern "C" void dropout_1d_backward(
  dtype id,
  void* _stream,
  DevicePropertiesWrapper _props,
  const void* y_grad,
  void* x_grad,
  len_t x_len,
  uint64_t* int_mask,
  float p
) {
  const auto stream = static_cast<cudaStream_t>(_stream);
  const auto props = DeviceProperties::unwrap(_props);

  switch (id) {
    case SINGLE: {
        return __dropout_1d_backward<f32>(stream, props, y_grad, x_grad, x_len, int_mask, p);
    }
    case DOUBLE: {
        return __dropout_1d_backward<f64>(stream, props, y_grad, x_grad, x_len, int_mask, p);
    }
    default:
      SYSTEM_EXIT("Unsupported Data Type");
  }
}

#endif
