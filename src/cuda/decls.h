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

// Wrappers for zig interface
typedef struct { void* ptr; } CutensorWrapper;
typedef struct { void* ptr; } DevicePropertiesWrapper;
typedef struct { void* ptr; } GraphWrapper;

typedef unsigned char reduxtype;
static const reduxtype RDX_NONE = 0;
static const reduxtype RDX_MEAN = 1;
static const reduxtype RDX_SUM = 2;

typedef unsigned char smaxtype;
static const smaxtype SMAX_FAST = 0;
static const smaxtype SMAX_MAX = 1;
static const smaxtype SMAX_LOG = 2;

typedef unsigned char randtype;
static const randtype UNIFORM = 0;
static const randtype NORMAL = 1;

typedef enum { ADD, MIN, MAX, MUL } BINARY_OP;

#endif
