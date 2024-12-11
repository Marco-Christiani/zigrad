#ifndef __BLAS_GEMV_ZIG__
#define __BLAS_GEMV_ZIG__

#include "blas_utils.cu"

extern "C" void gemv(
    dtype id, 
    void* cublas_handle,
    const void* a_data, 
    const void* b_data, 
    void* c_data, 
    len_t m,
    len_t n,
    bool trans_a,
    double alpha,
    double beta
) {
  const int _m = static_cast<int>(m);
  const int _n = static_cast<int>(n);
  const auto _trans_a = (trans_a) ? CUBLAS_OP_T : CUBLAS_OP_N;
  
  switch (id) {
    case SINGLE: {
      const float _alpha = static_cast<float>(alpha);
      const float _beta = static_cast<float>(beta);
      return CUBLAS_ASSERT(cublasSgemv(
        get_handle(cublas_handle), 
        _trans_a,
        _n, _m,
        &_alpha,
        static_cast<const float*>(b_data), 1,
        static_cast<const float*>(a_data), _n,
        &_beta,
        static_cast<float*>(c_data), 1
      ));
    }
    case DOUBLE: {
      return CUBLAS_ASSERT(cublasDgemv(
        get_handle(cublas_handle), 
        _trans_a,
        _n, _m,
        &alpha,
        static_cast<const double*>(b_data), 1,
        static_cast<const double*>(a_data), _n,
        &beta,
        static_cast<double*>(c_data), 1
      ));
    }
  }
}

#endif
