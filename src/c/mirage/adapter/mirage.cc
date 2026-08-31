#include "zigrad/mirage.h"

#include "mirage/kernel/device_memory_manager.h"
#include "mirage/kernel/graph.h"
#include "mirage/layout.h"
#include "mirage/search/search_c.h"
#include "mirage/transpiler/transpile.h"
#include "mirage/type.h"

#include <cuda_runtime_api.h>

#include <climits>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <utility>
#include <vector>

namespace {

thread_local std::string last_error;
std::mutex mirage_mutex;
int configured_device = -1;

void clear_error() { last_error.clear(); }

zg_mirage_status_t fail(zg_mirage_status_t status, char const *message) {
  last_error = message;
  return status;
}

template <typename F> zg_mirage_status_t protect(F &&fn) noexcept {
  clear_error();
  try {
    std::lock_guard<std::mutex> lock(mirage_mutex);
    return fn();
  } catch (std::bad_alloc const &) {
    return fail(ZG_MIRAGE_STATUS_OUT_OF_MEMORY, "allocation failed");
  } catch (std::exception const &error) {
    last_error = error.what();
    return ZG_MIRAGE_STATUS_EXTERNAL_FAILURE;
  } catch (...) {
    return fail(ZG_MIRAGE_STATUS_EXTERNAL_FAILURE,
                "Mirage raised an unknown exception");
  }
}

mirage::type::DataType to_mirage_dtype(zg_mirage_dtype_t dtype) {
  switch (dtype) {
  case ZG_MIRAGE_DTYPE_F16:
    return mirage::type::DT_FLOAT16;
  case ZG_MIRAGE_DTYPE_BF16:
    return mirage::type::DT_BFLOAT16;
  case ZG_MIRAGE_DTYPE_F32:
    return mirage::type::DT_FLOAT32;
  case ZG_MIRAGE_DTYPE_F64:
    return mirage::type::DT_DOUBLE;
  default:
    return mirage::type::DT_UNKNOWN;
  }
}

bool to_mirage_unary(zg_mirage_unary_op_t op,
                     mirage::type::KNOperatorType *out) {
  switch (op) {
  case ZG_MIRAGE_UNARY_EXP:
    *out = mirage::type::KN_EXP_OP;
    return true;
  case ZG_MIRAGE_UNARY_SQRT:
    *out = mirage::type::KN_SQRT_OP;
    return true;
  case ZG_MIRAGE_UNARY_SILU:
    *out = mirage::type::KN_SILU_OP;
    return true;
  case ZG_MIRAGE_UNARY_GELU:
    *out = mirage::type::KN_GELU_OP;
    return true;
  case ZG_MIRAGE_UNARY_RELU:
    *out = mirage::type::KN_RELU_OP;
    return true;
  case ZG_MIRAGE_UNARY_LOG:
    *out = mirage::type::KN_LOG_OP;
    return true;
  default:
    return false;
  }
}

bool to_mirage_binary(zg_mirage_binary_op_t op,
                      mirage::type::KNOperatorType *out) {
  switch (op) {
  case ZG_MIRAGE_BINARY_ADD:
    *out = mirage::type::KN_ADD_OP;
    return true;
  case ZG_MIRAGE_BINARY_MUL:
    *out = mirage::type::KN_MUL_OP;
    return true;
  case ZG_MIRAGE_BINARY_DIV:
    *out = mirage::type::KN_DIV_OP;
    return true;
  case ZG_MIRAGE_BINARY_POW:
    *out = mirage::type::KN_POW_OP;
    return true;
  default:
    return false;
  }
}

char const *preset_name(zg_mirage_search_preset_t preset) {
  switch (preset) {
  case ZG_MIRAGE_SEARCH_DEFAULT:
    return nullptr;
  case ZG_MIRAGE_SEARCH_ATTENTION:
    return "attention";
  case ZG_MIRAGE_SEARCH_LORA:
    return "lora";
  case ZG_MIRAGE_SEARCH_MLP:
    return "mlp";
  default:
    return nullptr;
  }
}

bool options_size_valid(size_t actual, size_t expected) {
  return actual >= expected;
}

zg_mirage_arg_source_t
from_argument_source(mirage::transpiler::KernelArgumentSource source) {
  switch (source) {
  case mirage::transpiler::KernelArgumentSource::INPUT:
    return ZG_MIRAGE_ARG_INPUT;
  case mirage::transpiler::KernelArgumentSource::OUTPUT:
    return ZG_MIRAGE_ARG_OUTPUT;
  case mirage::transpiler::KernelArgumentSource::WORKSPACE:
    return ZG_MIRAGE_ARG_WORKSPACE;
  }
  return ZG_MIRAGE_ARG_WORKSPACE;
}

} // namespace

struct zg_mirage_session {
  int32_t device_ordinal;
  int32_t target_cc;
};

struct zg_mirage_graph {
  std::unique_ptr<mirage::kernel::Graph> graph;
  std::vector<mirage::kernel::DTensor> tensors;
  std::vector<std::vector<size_t>> input_strides;
  int32_t target_cc = 80;

  zg_mirage_graph()
      : graph(std::make_unique<mirage::kernel::Graph>(
            dim3(1, 1, 1), /*disable_fingerprint=*/true)) {}

  explicit zg_mirage_graph(mirage::kernel::Graph *raw) : graph(raw) {}
};

struct zg_mirage_source {
  mirage::transpiler::TranspileResult result;
};

extern "C" {

uint32_t zg_mirage_abi_version(void) { return ZG_MIRAGE_ABI_VERSION; }

uint64_t zg_mirage_capabilities(void) {
  return ZG_MIRAGE_CAP_SYMBOLIC_OPTIMIZE | ZG_MIRAGE_CAP_CUDA_TRANSPILE |
         ZG_MIRAGE_CAP_EXPLICIT_CHECKPOINT | ZG_MIRAGE_CAP_LAUNCH_METADATA;
}

char const *zg_mirage_status_string(zg_mirage_status_t status) {
  switch (status) {
  case ZG_MIRAGE_STATUS_OK:
    return "ok";
  case ZG_MIRAGE_STATUS_INVALID_ARGUMENT:
    return "invalid_argument";
  case ZG_MIRAGE_STATUS_UNSUPPORTED:
    return "unsupported";
  case ZG_MIRAGE_STATUS_NOT_FOUND:
    return "not_found";
  case ZG_MIRAGE_STATUS_EXTERNAL_FAILURE:
    return "external_failure";
  case ZG_MIRAGE_STATUS_OUT_OF_MEMORY:
    return "out_of_memory";
  case ZG_MIRAGE_STATUS_ABI_MISMATCH:
    return "abi_mismatch";
  default:
    return "unknown_status";
  }
}

char const *zg_mirage_last_error(void) { return last_error.c_str(); }

zg_mirage_status_t zg_mirage_session_create(int32_t device_ordinal,
                                            zg_mirage_session_t **out) {
  return protect([&] {
    if (out == nullptr || device_ordinal < 0) {
      return fail(ZG_MIRAGE_STATUS_INVALID_ARGUMENT,
                  "invalid Mirage session arguments");
    }
    *out = nullptr;

    cudaDeviceProp properties{};
    cudaError_t error = cudaGetDeviceProperties(&properties, device_ordinal);
    if (error != cudaSuccess) {
      return fail(ZG_MIRAGE_STATUS_EXTERNAL_FAILURE, cudaGetErrorString(error));
    }

    if (configured_device == -1) {
      if (mirage::kernel::DeviceMemoryManager::singleton != nullptr) {
        return fail(ZG_MIRAGE_STATUS_EXTERNAL_FAILURE,
                    "Mirage device memory was initialized outside the adapter");
      }
      mirage::kernel::cython_set_gpu_device_id(device_ordinal);
      configured_device = device_ordinal;
    } else if (configured_device != device_ordinal) {
      return fail(ZG_MIRAGE_STATUS_UNSUPPORTED,
                  "Mirage supports one device ordinal per process");
    }

    error = cudaSetDevice(device_ordinal);
    if (error != cudaSuccess) {
      return fail(ZG_MIRAGE_STATUS_EXTERNAL_FAILURE, cudaGetErrorString(error));
    }

    auto session = std::make_unique<zg_mirage_session_t>();
    session->device_ordinal = device_ordinal;
    session->target_cc = properties.major * 10 + properties.minor;
    *out = session.release();
    return ZG_MIRAGE_STATUS_OK;
  });
}

void zg_mirage_session_destroy(zg_mirage_session_t *session) { delete session; }

zg_mirage_status_t zg_mirage_graph_create(zg_mirage_graph_t **out) {
  return protect([&] {
    if (out == nullptr) {
      return fail(ZG_MIRAGE_STATUS_INVALID_ARGUMENT,
                  "graph output pointer is null");
    }
    *out = nullptr;
    auto graph = std::make_unique<zg_mirage_graph_t>();
    *out = graph.release();
    return ZG_MIRAGE_STATUS_OK;
  });
}

void zg_mirage_graph_destroy(zg_mirage_graph_t *graph) { delete graph; }

zg_mirage_status_t
zg_mirage_graph_new_input(zg_mirage_graph_t *graph,
                          zg_mirage_tensor_spec_t const *spec,
                          zg_mirage_tensor_t *out) {
  return protect([&] {
    if (graph == nullptr || spec == nullptr || out == nullptr ||
        spec->rank == 0 || spec->rank > ZG_MIRAGE_MAX_RANK) {
      return fail(ZG_MIRAGE_STATUS_INVALID_ARGUMENT,
                  "invalid graph input arguments");
    }

    mirage::type::DataType dtype = to_mirage_dtype(spec->dtype);
    if (dtype == mirage::type::DT_UNKNOWN) {
      return fail(ZG_MIRAGE_STATUS_UNSUPPORTED,
                  "tensor dtype is not supported by Mirage");
    }

    std::vector<int> shape(spec->rank);
    for (uint32_t i = 0; i < spec->rank; ++i) {
      if (spec->dims[i] <= 0 || spec->dims[i] > INT_MAX) {
        return fail(ZG_MIRAGE_STATUS_INVALID_ARGUMENT,
                    "tensor dimension is outside Mirage's range");
      }
      shape[i] = static_cast<int>(spec->dims[i]);
    }

    bool explicit_strides = false;
    for (uint32_t i = 0; i < spec->rank; ++i) {
      explicit_strides |= spec->strides[i] != 0;
    }

    std::vector<size_t> strides(spec->rank);
    if (explicit_strides) {
      for (uint32_t i = 0; i < spec->rank; ++i) {
        if (spec->strides[i] <= 0) {
          return fail(ZG_MIRAGE_STATUS_INVALID_ARGUMENT,
                      "explicit strides must be positive");
        }
        strides[i] = static_cast<size_t>(spec->strides[i]);
      }
    } else {
      size_t stride = 1;
      for (uint32_t i = spec->rank; i-- > 0;) {
        strides[i] = stride;
        if (stride > std::numeric_limits<size_t>::max() /
                         static_cast<size_t>(shape[i])) {
          return fail(ZG_MIRAGE_STATUS_INVALID_ARGUMENT,
                      "dense input strides overflow size_t");
        }
        stride *= static_cast<size_t>(shape[i]);
      }
    }

    mirage::kernel::KNOperator *op = graph->graph->create_input_op(
        shape, strides, dtype, mirage::layout::DmemRowMajor);
    if (op == nullptr || op->output_tensors.size() != 1) {
      delete op;
      return fail(ZG_MIRAGE_STATUS_EXTERNAL_FAILURE,
                  "Mirage did not create one input");
    }
    graph->graph->operators.push_back(op);
    graph->tensors.push_back(op->output_tensors[0]);
    graph->input_strides.push_back(strides);
    *out = static_cast<zg_mirage_tensor_t>(graph->tensors.size() - 1);
    return ZG_MIRAGE_STATUS_OK;
  });
}

zg_mirage_status_t zg_mirage_graph_matmul(zg_mirage_graph_t *graph,
                                          zg_mirage_tensor_t lhs,
                                          zg_mirage_tensor_t rhs,
                                          zg_mirage_tensor_t *out) {
  return protect([&] {
    if (graph == nullptr || out == nullptr || lhs >= graph->tensors.size() ||
        rhs >= graph->tensors.size()) {
      return fail(ZG_MIRAGE_STATUS_INVALID_ARGUMENT,
                  "invalid matmul arguments");
    }
    if (graph->tensors[lhs].num_dims < 2 || graph->tensors[rhs].num_dims < 2) {
      return fail(ZG_MIRAGE_STATUS_INVALID_ARGUMENT,
                  "matmul inputs must have rank two or greater");
    }
    mirage::kernel::KNOperator *op = graph->graph->create_matmul_op(
        graph->tensors[lhs], graph->tensors[rhs]);
    if (op == nullptr || op->output_tensors.size() != 1) {
      delete op;
      return fail(ZG_MIRAGE_STATUS_EXTERNAL_FAILURE,
                  "Mirage did not create one matmul output");
    }
    graph->graph->operators.push_back(op);
    graph->tensors.push_back(op->output_tensors[0]);
    *out = static_cast<zg_mirage_tensor_t>(graph->tensors.size() - 1);
    return ZG_MIRAGE_STATUS_OK;
  });
}

zg_mirage_status_t zg_mirage_graph_unary(zg_mirage_graph_t *graph,
                                         zg_mirage_unary_op_t kind,
                                         zg_mirage_tensor_t input,
                                         zg_mirage_tensor_t *out) {
  return protect([&] {
    if (graph == nullptr || out == nullptr || input >= graph->tensors.size()) {
      return fail(ZG_MIRAGE_STATUS_INVALID_ARGUMENT,
                  "invalid unary operation arguments");
    }
    mirage::type::KNOperatorType op_type{};
    if (!to_mirage_unary(kind, &op_type)) {
      return fail(ZG_MIRAGE_STATUS_UNSUPPORTED,
                  "unary operation is not supported by Mirage");
    }
    mirage::kernel::KNOperator *op =
        graph->graph->create_elementunary_op(graph->tensors[input], op_type);
    if (op == nullptr || op->output_tensors.size() != 1) {
      delete op;
      return fail(ZG_MIRAGE_STATUS_EXTERNAL_FAILURE,
                  "Mirage did not create one unary output");
    }
    graph->graph->operators.push_back(op);
    graph->tensors.push_back(op->output_tensors[0]);
    *out = static_cast<zg_mirage_tensor_t>(graph->tensors.size() - 1);
    return ZG_MIRAGE_STATUS_OK;
  });
}

zg_mirage_status_t zg_mirage_graph_binary(zg_mirage_graph_t *graph,
                                          zg_mirage_binary_op_t kind,
                                          zg_mirage_tensor_t lhs,
                                          zg_mirage_tensor_t rhs,
                                          zg_mirage_tensor_t *out) {
  return protect([&] {
    if (graph == nullptr || out == nullptr || lhs >= graph->tensors.size() ||
        rhs >= graph->tensors.size()) {
      return fail(ZG_MIRAGE_STATUS_INVALID_ARGUMENT,
                  "invalid binary operation arguments");
    }
    mirage::type::KNOperatorType op_type{};
    if (!to_mirage_binary(kind, &op_type)) {
      return fail(ZG_MIRAGE_STATUS_UNSUPPORTED,
                  "binary operation is not supported by Mirage");
    }
    mirage::kernel::KNOperator *op = graph->graph->create_elementbinary_op(
        graph->tensors[lhs], graph->tensors[rhs], op_type);
    if (op == nullptr || op->output_tensors.size() != 1) {
      delete op;
      return fail(ZG_MIRAGE_STATUS_EXTERNAL_FAILURE,
                  "Mirage did not create one binary output");
    }
    graph->graph->operators.push_back(op);
    graph->tensors.push_back(op->output_tensors[0]);
    *out = static_cast<zg_mirage_tensor_t>(graph->tensors.size() - 1);
    return ZG_MIRAGE_STATUS_OK;
  });
}

zg_mirage_status_t zg_mirage_graph_reduction(zg_mirage_graph_t *graph,
                                             zg_mirage_tensor_t input,
                                             int32_t dim, int32_t factor,
                                             zg_mirage_tensor_t *out) {
  return protect([&] {
    if (graph == nullptr || out == nullptr || input >= graph->tensors.size() ||
        dim < 0 || factor <= 0) {
      return fail(ZG_MIRAGE_STATUS_INVALID_ARGUMENT,
                  "invalid reduction arguments");
    }
    mirage::kernel::KNOperator *op =
        graph->graph->create_reduction_op(graph->tensors[input], dim, factor);
    if (op == nullptr || op->output_tensors.size() != 1) {
      delete op;
      return fail(ZG_MIRAGE_STATUS_EXTERNAL_FAILURE,
                  "Mirage did not create one reduction output");
    }
    graph->graph->operators.push_back(op);
    graph->tensors.push_back(op->output_tensors[0]);
    *out = static_cast<zg_mirage_tensor_t>(graph->tensors.size() - 1);
    return ZG_MIRAGE_STATUS_OK;
  });
}

zg_mirage_status_t zg_mirage_graph_rms_norm(zg_mirage_graph_t *graph,
                                            zg_mirage_tensor_t input,
                                            int32_t normalized_size,
                                            zg_mirage_tensor_t *out) {
  return protect([&] {
    if (graph == nullptr || out == nullptr || input >= graph->tensors.size() ||
        normalized_size <= 0) {
      return fail(ZG_MIRAGE_STATUS_INVALID_ARGUMENT,
                  "invalid RMS normalization arguments");
    }
    mirage::kernel::KNOperator *op = graph->graph->create_rms_norm_op(
        graph->tensors[input], {normalized_size});
    if (op == nullptr || op->output_tensors.size() != 1) {
      delete op;
      return fail(ZG_MIRAGE_STATUS_EXTERNAL_FAILURE,
                  "Mirage did not create one RMS normalization output");
    }
    graph->graph->operators.push_back(op);
    graph->tensors.push_back(op->output_tensors[0]);
    *out = static_cast<zg_mirage_tensor_t>(graph->tensors.size() - 1);
    return ZG_MIRAGE_STATUS_OK;
  });
}

zg_mirage_status_t zg_mirage_graph_mark_output(zg_mirage_graph_t *graph,
                                               zg_mirage_tensor_t tensor) {
  return protect([&] {
    if (graph == nullptr || tensor >= graph->tensors.size()) {
      return fail(ZG_MIRAGE_STATUS_INVALID_ARGUMENT,
                  "invalid graph output handle");
    }

    mirage::kernel::DTensor const &dtensor = graph->tensors[tensor];
    std::vector<size_t> strides(dtensor.num_dims);
    size_t stride = 1;
    for (int i = dtensor.num_dims - 1; i >= 0; --i) {
      strides[i] = stride;
      if (dtensor.dim[i] <= 0 ||
          stride > std::numeric_limits<size_t>::max() /
                       static_cast<size_t>(dtensor.dim[i])) {
        return fail(ZG_MIRAGE_STATUS_EXTERNAL_FAILURE,
                    "Mirage output dimensions do not fit dense strides");
      }
      stride *= static_cast<size_t>(dtensor.dim[i]);
    }

    mirage::kernel::KNOperator *op =
        graph->graph->create_output_op(dtensor, strides);
    if (op == nullptr || !op->output_tensors.empty()) {
      delete op;
      return fail(ZG_MIRAGE_STATUS_EXTERNAL_FAILURE,
                  "Mirage did not create a valid output marker");
    }
    graph->graph->operators.push_back(op);
    return ZG_MIRAGE_STATUS_OK;
  });
}

zg_mirage_status_t
zg_mirage_optimize(zg_mirage_session_t *session, zg_mirage_graph_t const *graph,
                   zg_mirage_optimize_options_t const *options,
                   zg_mirage_graph_t **out) {
  return protect([&] {
    if (session == nullptr || graph == nullptr || out == nullptr) {
      return fail(ZG_MIRAGE_STATUS_INVALID_ARGUMENT,
                  "invalid optimize arguments");
    }
    *out = nullptr;
    if (options != nullptr &&
        !options_size_valid(options->struct_size, sizeof(*options))) {
      return fail(ZG_MIRAGE_STATUS_ABI_MISMATCH,
                  "optimize options are smaller than this ABI");
    }

    double time_limit = options != nullptr ? options->time_limit_seconds : 0.0;
    bool verbose = options != nullptr && options->verbose != 0;
    char const *checkpoint = nullptr;
    if (options != nullptr && options->checkpoint_path != nullptr &&
        options->checkpoint_path[0] != '\0') {
      checkpoint = options->checkpoint_path;
    }
    zg_mirage_search_preset_t preset =
        options != nullptr ? options->preset : ZG_MIRAGE_SEARCH_DEFAULT;
    char const *preset_string = preset_name(preset);
    if (preset != ZG_MIRAGE_SEARCH_DEFAULT && preset_string == nullptr) {
      return fail(ZG_MIRAGE_STATUS_INVALID_ARGUMENT,
                  "unknown Mirage search preset");
    }

    cudaError_t cuda_error = cudaSetDevice(session->device_ordinal);
    if (cuda_error != cudaSuccess) {
      return fail(ZG_MIRAGE_STATUS_EXTERNAL_FAILURE,
                  cudaGetErrorString(cuda_error));
    }

    mirage::kernel::Graph *raw = mirage::search_c::cython_search_symbolic(
        graph->graph.get(), checkpoint, verbose, preset_string, time_limit);
    if (raw == nullptr) {
      return fail(ZG_MIRAGE_STATUS_NOT_FOUND,
                  "Mirage found no valid optimized graph");
    }

    auto optimized = std::make_unique<zg_mirage_graph_t>(raw);
    optimized->target_cc = session->target_cc;
    for (mirage::kernel::KNOperator const *op : raw->operators) {
      if (op->op_type == mirage::type::KN_INPUT_OP) {
        auto const *input = static_cast<mirage::kernel::KNInputOp const *>(op);
        optimized->input_strides.push_back(input->input_strides);
      }
    }
    *out = optimized.release();
    return ZG_MIRAGE_STATUS_OK;
  });
}

zg_mirage_status_t
zg_mirage_transpile(zg_mirage_graph_t const *graph,
                    zg_mirage_transpile_options_t const *options,
                    zg_mirage_source_t **out) {
  return protect([&] {
    if (graph == nullptr || out == nullptr) {
      return fail(ZG_MIRAGE_STATUS_INVALID_ARGUMENT,
                  "invalid transpile arguments");
    }
    *out = nullptr;
    if (options != nullptr &&
        !options_size_valid(options->struct_size, sizeof(*options))) {
      return fail(ZG_MIRAGE_STATUS_ABI_MISMATCH,
                  "transpile options are smaller than this ABI");
    }

    mirage::transpiler::TranspilerConfig config{};
    config.target_cc = options != nullptr && options->target_cc > 0
                           ? options->target_cc
                           : graph->target_cc;
    config.profiling = false;
    config.num_consumer_wgs = 2;
    config.num_producer_wgs = 1;
    config.pipeline_stages = options != nullptr && options->pipeline_stages > 0
                                 ? options->pipeline_stages
                                 : 2;
    config.enable_online_softmax = false;

    mirage::transpiler::TranspileResult result = mirage::transpiler::transpile(
        graph->graph.get(), config, graph->input_strides);
    if (result.error_type != mirage::transpiler::CUDA_T_SUCCESS) {
      return fail(ZG_MIRAGE_STATUS_UNSUPPORTED,
                  "Mirage could not transpile the optimized graph");
    }

    auto source = std::make_unique<zg_mirage_source_t>(
        zg_mirage_source_t{std::move(result)});
    *out = source.release();
    return ZG_MIRAGE_STATUS_OK;
  });
}

void zg_mirage_source_destroy(zg_mirage_source_t *source) { delete source; }

char const *zg_mirage_source_code(zg_mirage_source_t const *source) {
  return source != nullptr ? source->result.code.c_str() : "";
}

size_t zg_mirage_source_code_len(zg_mirage_source_t const *source) {
  return source != nullptr ? source->result.code.size() : 0;
}

size_t zg_mirage_source_workspace_size(zg_mirage_source_t const *source) {
  return source != nullptr ? source->result.buf_size : 0;
}

size_t zg_mirage_source_num_kernels(zg_mirage_source_t const *source) {
  return source != nullptr ? source->result.kernel_directives.size() : 0;
}

zg_mirage_status_t
zg_mirage_source_kernel_meta(zg_mirage_source_t const *source,
                             size_t kernel_index,
                             zg_mirage_kernel_meta_t *out) {
  return protect([&] {
    if (source == nullptr || out == nullptr ||
        kernel_index >= source->result.kernel_directives.size()) {
      return fail(ZG_MIRAGE_STATUS_INVALID_ARGUMENT,
                  "invalid kernel metadata index");
    }
    auto const &kernel = source->result.kernel_directives[kernel_index];
    out->function_name = kernel.function_name.c_str();
    out->function_name_len = kernel.function_name.size();
    out->shared_memory_bytes = kernel.smem_size;
    for (size_t i = 0; i < 3; ++i) {
      out->grid_dim[i] = kernel.grid_dim[i];
      out->block_dim[i] = kernel.block_dim[i];
    }
    return ZG_MIRAGE_STATUS_OK;
  });
}

size_t zg_mirage_source_kernel_num_args(zg_mirage_source_t const *source,
                                        size_t kernel_index) {
  if (source == nullptr ||
      kernel_index >= source->result.kernel_directives.size()) {
    return 0;
  }
  return source->result.kernel_directives[kernel_index].arguments.size();
}

zg_mirage_status_t zg_mirage_source_kernel_arg(zg_mirage_source_t const *source,
                                               size_t kernel_index,
                                               size_t arg_index,
                                               zg_mirage_kernel_arg_t *out) {
  return protect([&] {
    if (source == nullptr || out == nullptr ||
        kernel_index >= source->result.kernel_directives.size() ||
        arg_index >=
            source->result.kernel_directives[kernel_index].arguments.size()) {
      return fail(ZG_MIRAGE_STATUS_INVALID_ARGUMENT,
                  "invalid kernel argument index");
    }
    auto const &argument =
        source->result.kernel_directives[kernel_index].arguments[arg_index];
    out->source = from_argument_source(argument.source);
    out->index_or_offset = argument.index_or_offset;
    return ZG_MIRAGE_STATUS_OK;
  });
}

} // extern "C"
