#ifndef __DEVICE_UTILS_ZIG_H__
#define __DEVICE_UTILS_ZIG_H__

#include "decls.h"

EXTERN_C void initDevice(unsigned);
EXTERN_C void* memAlloc(len_t N, void*);
EXTERN_C void memcpyHtoD(void* dptr, void const* hptr, len_t N, void*);
EXTERN_C void memcpyDtoH(void* hptr, void const* dptr, len_t N, void*);
EXTERN_C void memcpyDtoD(void* hptr, void const* dptr, len_t N, void*);
EXTERN_C void memFill(dtype id, void* data, len_t n, const void* value, void* stream);
EXTERN_C void memSequence(dtype id, void* data, len_t n, const void* init, const void* step, void* stream);
EXTERN_C void memFree(void* dptr, void*);
EXTERN_C void deviceSynchronize();
EXTERN_C void streamSynchronize(void*);
EXTERN_C void* initStream();
EXTERN_C void deinitStream(void*);
EXTERN_C void checkLastError();
EXTERN_C len_t deviceTotalMemory(unsigned);
EXTERN_C void* initCublasHandle(void* stream);
EXTERN_C void deinitCublasHandle(void* handle);
EXTERN_C void* initCudnnHandle(void* stream);
EXTERN_C void deinitCudnnHandle(void* handle);

#endif
