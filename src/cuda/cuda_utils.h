#ifndef __DEVICE_UTILS_ZIG_H__
#define __DEVICE_UTILS_ZIG_H__

#include "decls.h"

// general utilities
EXTERN_C void* mem_alloc(len_t N, StreamWrapper);
EXTERN_C void memcpy_HtoD(void* dptr, void const* hptr, len_t N, StreamWrapper);
EXTERN_C void memcpy_DtoH(void* hptr, void const* dptr, len_t N, StreamWrapper);
EXTERN_C void memcpy_DtoD(void* hptr, void const* dptr, len_t N, StreamWrapper);
EXTERN_C void mem_fill(dtype id, void* data, len_t n, const void* value, StreamWrapper);
EXTERN_C void mem_sequence(dtype id, void* data, len_t n, const void* init, const void* step, StreamWrapper);
EXTERN_C void mem_random(dtype id, void* x, len_t n, randtype op, unsigned seed, StreamWrapper);
EXTERN_C void mem_free(void* dptr, StreamWrapper);
EXTERN_C void mem_take(dtype id, const void* src, len_t src_len, const len_t* idxs, len_t idxs_len, void* dst, StreamWrapper);

// device api
EXTERN_C DevicePropertiesWrapper init_device(unsigned);
EXTERN_C void check_last_error();
EXTERN_C void device_synchronize();
EXTERN_C len_t device_total_memory(unsigned);

// stream api
EXTERN_C StreamWrapper init_stream();
EXTERN_C void deinit_stream(StreamWrapper);
EXTERN_C void stream_synchronize(StreamWrapper);

// cublas api
EXTERN_C CublasWrapper init_cublas_handle(StreamWrapper);
EXTERN_C void deinit_cublas_handle(CublasWrapper);

// cudnn api
EXTERN_C CudnnWrapper init_cudnn_handle(StreamWrapper);
EXTERN_C void deinit_cudnn_handle(CudnnWrapper);

// cutensor api
EXTERN_C CutensorWrapper init_cutensor_handle(StreamWrapper);
EXTERN_C void deinit_cutensor_handle(CutensorWrapper);

// stream capture api
EXTERN_C GraphWrapper open_capture(StreamWrapper);
EXTERN_C void save_capture(GraphWrapper wrapper, StreamWrapper stream);
EXTERN_C void free_capture(GraphWrapper wrapper, StreamWrapper stream);
EXTERN_C void run_capture(GraphWrapper wrapper, StreamWrapper stream);

#endif
