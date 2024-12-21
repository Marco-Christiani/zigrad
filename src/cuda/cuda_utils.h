#ifndef __DEVICE_UTILS_ZIG_H__
#define __DEVICE_UTILS_ZIG_H__

#include "decls.h"

EXTERN_C void init_device(unsigned);
EXTERN_C void* mem_alloc(len_t N, void*);
EXTERN_C void memcpy_HtoD(void* dptr, void const* hptr, len_t N, void*);
EXTERN_C void memcpy_DtoH(void* hptr, void const* dptr, len_t N, void*);
EXTERN_C void memcpy_DtoD(void* hptr, void const* dptr, len_t N, void*);
EXTERN_C void mem_fill(dtype id, void* data, len_t n, const void* value, void* stream);
EXTERN_C void mem_sequence(dtype id, void* data, len_t n, const void* init, const void* step, void* stream);
EXTERN_C void mem_random(dtype id, void* x, len_t n, randtype op, unsigned seed, void* stream);
EXTERN_C void mem_free(void* dptr, void*);
EXTERN_C void device_synchronize();
EXTERN_C void stream_synchronize(void*);
EXTERN_C void* init_stream();
EXTERN_C void deinit_stream(void*);
EXTERN_C void check_last_error();
EXTERN_C len_t device_total_memory(unsigned);
EXTERN_C void* init_cublas_handle(void* stream);
EXTERN_C void deinit_cublas_handle(void* handle);
EXTERN_C void* init_cudnn_handle(void* stream);
EXTERN_C void deinit_cudnn_handle(void* handle);
EXTERN_C CutensorWrapper init_cutensor_handle(void* stream);
EXTERN_C void deinit_cutensor_handle(CutensorWrapper);

#endif
