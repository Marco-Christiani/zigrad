#ifndef __BLAS_CMP_REDUX_REV_ZIG__
#define __BLAS_CMP_REDUX_REV_ZIG__

#include "blas_utils.cu"

// TODO: give this a proper include structure
#include "../device_properties.cu"

template<typename T>
__device__ void __project_scalar(
    T value,
    T grad,
    const T* __restrict__ y_vals,
          T* __restrict__ y_grad,
    len_t chunk_size 
) {
    for (len_t idx = threadIdx.x; idx < chunk_size; idx += blockDim.x) {
      if (value == y_vals[idx]) y_grad[idx] += grad;
    }
}

// simple optimization: copy src to shared memory and then copy across
// boundaries before sliding the window to the right.
template<typename T>
__device__ void __project_tensor(
    const T* __restrict__ x_vals, 
    const T* __restrict__ x_grad, 
    const T* __restrict__ y_vals,
          T* __restrict__ y_grad,
    len_t chunk_size 
) {
    for (len_t idx = threadIdx.x; idx < chunk_size; idx += blockDim.x) {
      if (x_vals[idx] == y_vals[idx]) y_grad[idx] += x_grad[idx];
    }
}

// technically, if the lengths are ever the same,
// then this should always return true.
__device__ bool __equal(
  const char* a_syms, len_t a_len, len_t a_pos,
  const char* b_syms, len_t b_len, len_t b_pos
) {
  if (a_len != b_len)
    return false;

  for (; a_pos < a_len; ++a_pos, ++b_pos) {
    if (a_syms[a_pos] != b_syms[b_pos]) return false;
  }

  return true;
}

__device__ bool __contains(
  len_t a_sym, const char* b_syms, len_t b_len, len_t b_pos
) {
  for (; b_pos < b_len; ++b_pos) {
    if (a_sym == b_syms[b_pos]) return true;
  }
  return false;
}

typedef struct {
  len_t sizes[8];
  len_t strides[8];
  char symbols[8];
  len_t len;
} Modes;

template<typename T>
__device__ void __cmp_reduce_bwds_rec(
  const T* __restrict__ x_vals, const Modes& x_modes, len_t x_pos, len_t x_offset,
  const T* __restrict__ x_grad,
  const T* __restrict__ y_vals, const Modes& y_modes, len_t y_pos, len_t y_offset,
        T* __restrict__ y_grad
) {
  if (__equal(
    x_modes.symbols, x_modes.len, x_pos,
    y_modes.symbols, y_modes.len, y_pos + 1
  )) {
    const len_t num_chunks = y_modes.sizes[y_pos];
    const len_t chunk_size = y_modes.strides[y_pos];
    // divide tensor copying work between blocks
    for (len_t i = blockIdx.x; i < num_chunks; i += gridDim.x) {
      __project_tensor<T>(
        (x_vals + x_offset) + (i * chunk_size),
        (x_grad + x_offset) + (i * chunk_size),
        (y_vals + y_offset) + (i * chunk_size),
        (y_grad + y_offset) + (i * chunk_size),
        chunk_size
      );
    }
  }

  if (x_pos + 1 == x_modes.len and not __contains(x_modes.symbols[x_pos], y_modes.symbols, y_modes.len, y_pos + 1)) {
    const len_t num_chunks = y_modes.sizes[y_pos];
    const len_t chunk_size = y_modes.strides[y_pos];
    const auto src_offset = x_vals + x_offset;
    const auto grd_offset = x_vals + x_offset;
    for (len_t i = blockIdx.x; i < num_chunks; i += gridDim.x) {
      __project_scalar<T>(
        src_offset[i],
        grd_offset[i],
        (y_vals + y_offset) + (i * chunk_size),
        (y_grad + y_offset) + (i * chunk_size),
        chunk_size
      );
    }
  }

  if (x_modes.symbols[x_pos] == y_modes.symbols[y_pos]) {
    for (len_t i = 0; i < y_modes.sizes[y_pos]; ++i) {
      __cmp_reduce_bwds_rec(
          x_vals, x_modes, x_pos + 1, x_offset + i * x_modes.strides[x_pos],
          x_grad,
          y_vals, y_modes, y_pos + 1, y_offset + i * y_modes.strides[y_pos],
          y_grad
      );
    }
  } else {
    for (len_t i = 0; i < y_modes.sizes[y_pos]; ++i) {
      __cmp_reduce_bwds_rec(
          x_vals, x_modes, x_pos, x_offset,
          x_grad,
          y_vals, y_modes, y_pos + 1, y_offset + i * y_modes.strides[y_pos],
          y_grad
      );
    }
  }
}

template<typename T>
__global__ void __cmp_reduce_bwds(
  const void* __restrict__ x_vals, Modes x_modes,
  const void* __restrict__ x_grad,
  const void* __restrict__ y_vals, Modes y_modes,
        void* __restrict__ y_grad
) {
  __cmp_reduce_bwds_rec(
    static_cast<const T*>(x_vals), x_modes, 0, 0,
    static_cast<const T*>(x_grad),
    static_cast<const T*>(y_vals), y_modes, 0, 0,
    static_cast<T*>(y_grad)
  );
}


// this function is a little excesive for its setup
// but will make more pratical sense for einsum
extern "C" void cmp_reduce_bwds(  
  dtype id,
  void* stream,
  DevicePropertiesWrapper pwrap,
  const void* x_vals, const len_t* x_sizes, const len_t* x_strides, const len_t x_dim_len,
        void* x_grad,
  const void* y_vals, const len_t* y_sizes, const len_t* y_strides, const len_t y_dim_len,
  const void* y_grad,
  const len_t* reduce_idxs, len_t reduce_idxs_len
) {
  const DeviceProperties* prop = DeviceProperties::unwrap(pwrap);

  int requested_blocks = 1;
  const int first_dif = (int)reduce_idxs[reduce_idxs_len - 1];
  for (int i = (int)x_dim_len - 1; i != first_dif and i >= 0; --i) {
    requested_blocks *= x_sizes[i];
  }

  {
    // try to launch as many blocks as possible
    const int thread_limit = std::min(
      requested_blocks * prop->max_threads_per_block,
      prop->multi_processor_count * prop->max_threads_per_multi_processor
    );

    // TODO: max threads doesn't always give the best performance
    requested_blocks = std::min(prop->grid_sizes.x, thread_limit / prop->max_threads_per_block);
  }

  Modes x_modes;
  Modes y_modes;
  memcpy(&x_modes.sizes[0], x_sizes, x_dim_len * sizeof(len_t));
  memcpy(&y_modes.sizes[0], y_sizes, y_dim_len * sizeof(len_t));
  memcpy(&x_modes.sizes[0], x_strides, x_dim_len * sizeof(len_t));
  memcpy(&y_modes.sizes[0], y_strides, y_dim_len * sizeof(len_t));
  x_modes.len = x_dim_len;
  y_modes.len = y_dim_len;

  {
    len_t x_i = 0;
    len_t y_i = 0;
    len_t r_i = 0;
    char sym = 'i';
    while (x_i < x_dim_len) {
      x_modes.symbols[x_i] = sym;

      if (r_i < reduce_idxs_len and x_i == reduce_idxs[r_i]) {
        r_i += 1;
        continue;
      }

      if (y_i < y_dim_len) {
        y_modes.symbols[y_i] = sym;
      }

      x_i += 1;
      y_i += 1;
      sym += 1;
    }
  }

  // I accidentally wrote this in terms of a broadcast - the arguments are reversed...
  switch (id) {
    case SINGLE:
      __cmp_reduce_bwds<f32><<<requested_blocks, prop->max_threads_per_block>>>(y_vals, y_modes, y_grad, x_vals, x_modes, x_grad);
      break;
    case DOUBLE:
      __cmp_reduce_bwds<f64><<<requested_blocks, prop->max_threads_per_block>>>(y_vals, y_modes, y_grad, x_vals, x_modes, x_grad);
      break;
    default:
      SYSTEM_EXIT("Unsupported data type");
  }

  CUDA_ASSERT(cudaPeekAtLastError());
}




#endif
