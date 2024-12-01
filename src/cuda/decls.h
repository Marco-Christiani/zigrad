#ifndef __CUDA_DECLS_ZIG__
#define __CUDA_DECLS_ZIG__

#if defined(__cplusplus)
    #define EXTERN_C extern "C"
#else
    #define EXTERN_C extern
#endif

#include <stdbool.h> // bool
typedef unsigned long len_t;
typedef unsigned char dtype;
static const dtype SINGLE = 0;
static const dtype DOUBLE = 1;

#endif
