#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
  ZG_MIRAGE_ABI_VERSION = 1,
  ZG_MIRAGE_MAX_RANK = 4,
};

typedef enum zg_mirage_status {
  ZG_MIRAGE_STATUS_OK = 0,
  ZG_MIRAGE_STATUS_INVALID_ARGUMENT = 1,
  ZG_MIRAGE_STATUS_UNSUPPORTED = 2,
  ZG_MIRAGE_STATUS_NOT_FOUND = 3,
  ZG_MIRAGE_STATUS_EXTERNAL_FAILURE = 4,
  ZG_MIRAGE_STATUS_OUT_OF_MEMORY = 5,
  ZG_MIRAGE_STATUS_ABI_MISMATCH = 6,
} zg_mirage_status_t;

typedef enum zg_mirage_capability {
  ZG_MIRAGE_CAP_SYMBOLIC_OPTIMIZE = 1u << 0,
  ZG_MIRAGE_CAP_CUDA_TRANSPILE = 1u << 1,
  ZG_MIRAGE_CAP_EXPLICIT_CHECKPOINT = 1u << 2,
  ZG_MIRAGE_CAP_LAUNCH_METADATA = 1u << 3,
} zg_mirage_capability_t;

/// Returns the adapter ABI version.
uint32_t zg_mirage_abi_version(void);

/// Returns the adapter capabilities as a bitfield of `zg_mirage_capability_t`.
uint64_t zg_mirage_capabilities(void);

/// Returns a stable name for `status`.
char const *zg_mirage_status_string(zg_mirage_status_t status);

/// Returns the current thread's last adapter diagnostic.
///
/// The returned pointer remains valid until the next status-returning adapter
///  call on the same thread.
char const *zg_mirage_last_error(void);

typedef enum zg_mirage_dtype {
  ZG_MIRAGE_DTYPE_F16 = 0,
  ZG_MIRAGE_DTYPE_BF16 = 1,
  ZG_MIRAGE_DTYPE_F32 = 2,
  ZG_MIRAGE_DTYPE_F64 = 3,
} zg_mirage_dtype_t;

typedef enum zg_mirage_unary_op {
  ZG_MIRAGE_UNARY_EXP = 0,
  ZG_MIRAGE_UNARY_SQRT = 1,
  ZG_MIRAGE_UNARY_SILU = 2,
  ZG_MIRAGE_UNARY_GELU = 3,
  ZG_MIRAGE_UNARY_RELU = 4,
  ZG_MIRAGE_UNARY_LOG = 5,
} zg_mirage_unary_op_t;

typedef enum zg_mirage_binary_op {
  ZG_MIRAGE_BINARY_ADD = 0,
  ZG_MIRAGE_BINARY_MUL = 1,
  ZG_MIRAGE_BINARY_DIV = 2,
  ZG_MIRAGE_BINARY_POW = 3,
} zg_mirage_binary_op_t;

typedef uint32_t zg_mirage_tensor_t;

typedef struct zg_mirage_tensor_spec {
  /// Tensor element type.
  zg_mirage_dtype_t dtype;

  /// Number of active entries in `dims` and `strides`.
  uint32_t rank;

  /// Positive logical dimensions.
  int64_t dims[ZG_MIRAGE_MAX_RANK];

  /// Element strides.
  ///
  /// All-zero strides request a dense row-major layout.
  int64_t strides[ZG_MIRAGE_MAX_RANK];
} zg_mirage_tensor_spec_t;

typedef struct zg_mirage_session zg_mirage_session_t;
typedef struct zg_mirage_graph zg_mirage_graph_t;
typedef struct zg_mirage_source zg_mirage_source_t;

/// Creates a Mirage session for `device_ordinal`.
///
/// Adapter calls serialize access to Mirage's process-global state, and a
///  process may use one device ordinal.
zg_mirage_status_t zg_mirage_session_create(int32_t device_ordinal,
                                            zg_mirage_session_t **out);

/// Destroys `session`.
void zg_mirage_session_destroy(zg_mirage_session_t *session);

/// Creates an empty graph with fingerprint allocation disabled.
zg_mirage_status_t zg_mirage_graph_create(zg_mirage_graph_t **out);

/// Destroys `graph`.
void zg_mirage_graph_destroy(zg_mirage_graph_t *graph);

/// Adds an input tensor and returns its graph-local handle.
zg_mirage_status_t
zg_mirage_graph_new_input(zg_mirage_graph_t *graph,
                          zg_mirage_tensor_spec_t const *spec,
                          zg_mirage_tensor_t *out);

/// Adds a matrix multiplication operation.
zg_mirage_status_t zg_mirage_graph_matmul(zg_mirage_graph_t *graph,
                                          zg_mirage_tensor_t lhs,
                                          zg_mirage_tensor_t rhs,
                                          zg_mirage_tensor_t *out);

/// Adds an elementwise unary operation.
zg_mirage_status_t zg_mirage_graph_unary(zg_mirage_graph_t *graph,
                                         zg_mirage_unary_op_t op,
                                         zg_mirage_tensor_t input,
                                         zg_mirage_tensor_t *out);

/// Adds an elementwise binary operation.
zg_mirage_status_t zg_mirage_graph_binary(zg_mirage_graph_t *graph,
                                          zg_mirage_binary_op_t op,
                                          zg_mirage_tensor_t lhs,
                                          zg_mirage_tensor_t rhs,
                                          zg_mirage_tensor_t *out);

/// Adds a reduction operation.
zg_mirage_status_t zg_mirage_graph_reduction(zg_mirage_graph_t *graph,
                                             zg_mirage_tensor_t input,
                                             int32_t dim, int32_t factor,
                                             zg_mirage_tensor_t *out);

/// Adds an RMS normalization operation.
zg_mirage_status_t zg_mirage_graph_rms_norm(zg_mirage_graph_t *graph,
                                            zg_mirage_tensor_t input,
                                            int32_t normalized_size,
                                            zg_mirage_tensor_t *out);

/// Marks `tensor` as a graph output.
zg_mirage_status_t zg_mirage_graph_mark_output(zg_mirage_graph_t *graph,
                                               zg_mirage_tensor_t tensor);

typedef enum zg_mirage_search_preset {
  ZG_MIRAGE_SEARCH_DEFAULT = 0,
  ZG_MIRAGE_SEARCH_ATTENTION = 1,
  ZG_MIRAGE_SEARCH_LORA = 2,
} zg_mirage_search_preset_t;

typedef struct zg_mirage_optimize_options {
  /// Set to `sizeof(zg_mirage_optimize_options_t)`.
  size_t struct_size;

  /// Search limit in seconds.
  ///
  /// A non-positive value uses Mirage's default.
  double time_limit_seconds;

  /// Optional null-terminated checkpoint path.
  ///
  /// Null or empty disables checkpoint I/O.
  char const *checkpoint_path;

  /// Named search configuration.
  zg_mirage_search_preset_t preset;

  /// Nonzero enables upstream search diagnostics.
  uint8_t verbose;
} zg_mirage_optimize_options_t;

/// Runs Mirage symbolic optimization and returns one tuned graph.
///
/// A null or empty `checkpoint_path` disables checkpoint I/O.
zg_mirage_status_t
zg_mirage_optimize(zg_mirage_session_t *session, zg_mirage_graph_t const *graph,
                   zg_mirage_optimize_options_t const *options,
                   zg_mirage_graph_t **out);

typedef struct zg_mirage_transpile_options {
  /// Set to `sizeof(zg_mirage_transpile_options_t)`.
  size_t struct_size;

  /// CUDA compute capability times ten.
  ///
  /// A non-positive value uses the session device.
  int32_t target_cc;

  /// Pipeline depth.
  ///
  /// A non-positive value uses the adapter default.
  int32_t pipeline_stages;
} zg_mirage_transpile_options_t;

/// Transpiles a tuned graph to CUDA source and launch metadata.
zg_mirage_status_t
zg_mirage_transpile(zg_mirage_graph_t const *graph,
                    zg_mirage_transpile_options_t const *options,
                    zg_mirage_source_t **out);

/// Destroys `source`.
void zg_mirage_source_destroy(zg_mirage_source_t *source);

/// Returns the null-terminated CUDA source.
char const *zg_mirage_source_code(zg_mirage_source_t const *source);

/// Returns the CUDA source length in bytes, excluding the null terminator.
size_t zg_mirage_source_code_len(zg_mirage_source_t const *source);

/// Returns the required workspace size in bytes.
size_t zg_mirage_source_workspace_size(zg_mirage_source_t const *source);

typedef enum zg_mirage_arg_source {
  ZG_MIRAGE_ARG_INPUT = 0,
  ZG_MIRAGE_ARG_OUTPUT = 1,
  ZG_MIRAGE_ARG_WORKSPACE = 2,
} zg_mirage_arg_source_t;

typedef struct zg_mirage_kernel_meta {
  /// Null-terminated function name.
  ///
  /// The pointer remains valid until its source is destroyed.
  char const *function_name;

  /// Function name length excluding the null terminator.
  size_t function_name_len;

  /// Dynamic shared-memory requirement in bytes.
  size_t shared_memory_bytes;

  /// CUDA launch grid dimensions.
  uint32_t grid_dim[3];

  /// CUDA launch block dimensions.
  uint32_t block_dim[3];
} zg_mirage_kernel_meta_t;

typedef struct zg_mirage_kernel_arg {
  /// Dispatch-time pointer source.
  zg_mirage_arg_source_t source;

  /// Input or output index, or byte offset into the workspace.
  size_t index_or_offset;
} zg_mirage_kernel_arg_t;

/// Returns the number of generated custom kernels.
size_t zg_mirage_source_num_kernels(zg_mirage_source_t const *source);

/// Returns exact launch metadata for a generated kernel.
zg_mirage_status_t
zg_mirage_source_kernel_meta(zg_mirage_source_t const *source,
                             size_t kernel_index, zg_mirage_kernel_meta_t *out);

/// Returns the number of pointer arguments for a generated kernel.
size_t zg_mirage_source_kernel_num_args(zg_mirage_source_t const *source,
                                        size_t kernel_index);

/// Returns the source of a generated kernel pointer argument.
zg_mirage_status_t zg_mirage_source_kernel_arg(zg_mirage_source_t const *source,
                                               size_t kernel_index,
                                               size_t arg_index,
                                               zg_mirage_kernel_arg_t *out);

#ifdef __cplusplus
}
#endif
