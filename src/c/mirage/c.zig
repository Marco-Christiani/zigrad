//! Mirage C ABI declarations.
//!
//! Types are imported from the layered C API headers (types.h, graph.h,
//! source.h, ir.h) via @cImport. Function calls go through dlsym since
//! mirage is a runtime plugin, not a link-time dependency.
const std = @import("std");

const log = std.log.scoped(.@"zg/mirage_cffi");

// ---------------------------------------------------------------------------
// Types from C headers
// ---------------------------------------------------------------------------

const C = @import("c-mirage");

// Re-export C types under cleaner names.

pub const MirageGraph = C.mirage_graph_t;
pub const MirageSearchResult = C.mirage_search_result_t;
pub const MirageDevice = C.mirage_device_t;
pub const MirageSource = C.mirage_source_t;
pub const MirageTBGraph = C.mirage_tbgraph_t;
pub const MirageTensor = C.mirage_tensor_t;

pub const max_rank = C.MIRAGE_MAX_RANK;

pub const MirageStatus = C.mirage_status_t;
pub const status_ok = C.MIRAGE_STATUS_OK;
pub const status_invalid_argument = C.MIRAGE_STATUS_INVALID_ARGUMENT;
pub const status_internal_error = C.MIRAGE_STATUS_INTERNAL_ERROR;
pub const status_unsupported = C.MIRAGE_STATUS_UNSUPPORTED;
pub const status_not_found = C.MIRAGE_STATUS_NOT_FOUND;

pub const MirageDType = C.mirage_dtype_t;
pub const dtype_f16 = C.MIRAGE_DTYPE_F16;
pub const dtype_bf16 = C.MIRAGE_DTYPE_BF16;
pub const dtype_f32 = C.MIRAGE_DTYPE_F32;
pub const dtype_f64 = C.MIRAGE_DTYPE_F64;

pub const MirageUnaryOp = C.mirage_unary_op_t;
pub const unary_exp = C.MIRAGE_UNARY_EXP;
pub const unary_sqrt = C.MIRAGE_UNARY_SQRT;
pub const unary_silu = C.MIRAGE_UNARY_SILU;
pub const unary_gelu = C.MIRAGE_UNARY_GELU;
pub const unary_relu = C.MIRAGE_UNARY_RELU;
pub const unary_log = C.MIRAGE_UNARY_LOG;

pub const MirageBinaryOp = C.mirage_binary_op_t;
pub const binary_add = C.MIRAGE_BINARY_ADD;
pub const binary_mul = C.MIRAGE_BINARY_MUL;
pub const binary_div = C.MIRAGE_BINARY_DIV;
pub const binary_pow = C.MIRAGE_BINARY_POW;

pub const TensorSpec = C.mirage_tensor_spec_t;
pub const SearchOptions = C.mirage_search_options_t;
pub const TranspileOptions = C.mirage_transpile_options_t;
pub const KernelMeta = C.mirage_kernel_meta_t;
pub const SourceTraits = C.mirage_source_traits_t;

pub const source_uses_host_libs = C.MIRAGE_SOURCE_USES_HOST_LIBS;
pub const source_has_kernels = C.MIRAGE_SOURCE_HAS_KERNELS;
pub const source_device_callable = C.MIRAGE_SOURCE_DEVICE_CALLABLE;

pub const ArgSource = C.mirage_arg_source_t;
pub const arg_input = C.MIRAGE_ARG_INPUT;
pub const arg_output = C.MIRAGE_ARG_OUTPUT;
pub const arg_buf = C.MIRAGE_ARG_BUF;
pub const KernelArg = C.mirage_kernel_arg_t;

// IR types
pub const KnOpType = C.mirage_kn_op_type_t;
pub const TbOpType = C.mirage_tb_op_type_t;
pub const STensorSpec = C.mirage_stensor_spec_t;
pub const TbInputInfo = C.mirage_tb_input_info_t;
pub const TbOutputInfo = C.mirage_tb_output_info_t;
pub const Epilogue = C.mirage_epilogue_t;

// ---------------------------------------------------------------------------
// Function pointer types for dlsym loading
// ---------------------------------------------------------------------------

// types.h
const FnStatusString = *const fn (MirageStatus) callconv(.c) [*:0]const u8;

// graph.h - graph building
const FnGraphCreate = *const fn (*?*MirageGraph) callconv(.c) MirageStatus;
const FnGraphDestroy = *const fn (?*MirageGraph) callconv(.c) void;
const FnGraphNewInput = *const fn (?*MirageGraph, *const TensorSpec, *MirageTensor) callconv(.c) MirageStatus;
const FnGraphMatmul = *const fn (?*MirageGraph, MirageTensor, MirageTensor, *MirageTensor) callconv(.c) MirageStatus;
const FnGraphUnary = *const fn (?*MirageGraph, MirageUnaryOp, MirageTensor, *MirageTensor) callconv(.c) MirageStatus;
const FnGraphBinary = *const fn (?*MirageGraph, MirageBinaryOp, MirageTensor, MirageTensor, *MirageTensor) callconv(.c) MirageStatus;
const FnGraphReduction = *const fn (?*MirageGraph, MirageTensor, i32, i32, *MirageTensor) callconv(.c) MirageStatus;
const FnGraphRmsNorm = *const fn (?*MirageGraph, MirageTensor, i32, *MirageTensor) callconv(.c) MirageStatus;
const FnGraphMarkOutput = *const fn (?*MirageGraph, MirageTensor) callconv(.c) MirageStatus;

// graph.h - device
const FnDeviceCreate = *const fn (i32, *?*MirageDevice) callconv(.c) MirageStatus;
const FnDeviceDestroy = *const fn (?*MirageDevice) callconv(.c) void;
const FnDeviceMemInfo = *const fn (?*const MirageDevice, *usize, *usize) callconv(.c) MirageStatus;

// graph.h - search
const FnSearch = *const fn (?*MirageDevice, ?*const MirageGraph, ?*const SearchOptions, *?*MirageSearchResult) callconv(.c) MirageStatus;
const FnSearchResultDestroy = *const fn (?*MirageSearchResult) callconv(.c) void;
const FnSearchResultCount = *const fn (?*const MirageSearchResult) callconv(.c) usize;
const FnSearchResultGet = *const fn (?*const MirageSearchResult, usize) callconv(.c) ?*const MirageGraph;

// source.h
const FnTranspile = *const fn (?*const MirageGraph, ?*const TranspileOptions, *?*MirageSource) callconv(.c) MirageStatus;
const FnSourceDestroy = *const fn (?*MirageSource) callconv(.c) void;
const FnSourceTraits = *const fn (?*const MirageSource) callconv(.c) SourceTraits;
const FnSourceCode = *const fn (?*const MirageSource) callconv(.c) ?[*:0]const u8;
const FnSourceCodeLen = *const fn (?*const MirageSource) callconv(.c) usize;
const FnSourceBufSize = *const fn (?*const MirageSource) callconv(.c) usize;
const FnSourceMaxSmem = *const fn (?*const MirageSource) callconv(.c) usize;
const FnSourceNumOutputs = *const fn (?*const MirageSource) callconv(.c) usize;
const FnSourceOutputSpec = *const fn (?*const MirageSource, usize, *TensorSpec) callconv(.c) MirageStatus;
const FnSourceNumKernels = *const fn (?*const MirageSource) callconv(.c) usize;
const FnSourceKernelMeta = *const fn (?*const MirageSource, usize, *KernelMeta) callconv(.c) MirageStatus;
const FnSourceKernelNumArgs = *const fn (?*const MirageSource, usize) callconv(.c) usize;
const FnSourceKernelArg = *const fn (?*const MirageSource, usize, usize, *KernelArg) callconv(.c) MirageStatus;

// ir.h - kernel graph walk
const FnIrNumOps = *const fn (?*const MirageGraph) callconv(.c) usize;
const FnIrOpType = *const fn (?*const MirageGraph, usize) callconv(.c) KnOpType;
const FnIrOpNumInputs = *const fn (?*const MirageGraph, usize) callconv(.c) usize;
const FnIrOpNumOutputs = *const fn (?*const MirageGraph, usize) callconv(.c) usize;
const FnIrOpInput = *const fn (?*const MirageGraph, usize, usize) callconv(.c) MirageTensor;
const FnIrOpOutput = *const fn (?*const MirageGraph, usize, usize) callconv(.c) MirageTensor;
const FnIrTensorSpec = *const fn (?*const MirageGraph, MirageTensor, *TensorSpec) callconv(.c) MirageStatus;

// ir.h - threadblock graph
const FnIrOpTbgraph = *const fn (?*const MirageGraph, usize, *?*const MirageTBGraph) callconv(.c) MirageStatus;
const FnIrTbgraphGridDim = *const fn (?*const MirageTBGraph, *[3]u32) callconv(.c) void;
const FnIrTbgraphBlockDim = *const fn (?*const MirageTBGraph, *[3]u32) callconv(.c) void;
const FnIrTbgraphForloopRange = *const fn (?*const MirageTBGraph) callconv(.c) i32;
const FnIrTbgraphReductionDimx = *const fn (?*const MirageTBGraph) callconv(.c) i32;
const FnIrTbgraphNumOps = *const fn (?*const MirageTBGraph) callconv(.c) usize;
const FnIrTbgraphOpType = *const fn (?*const MirageTBGraph, usize) callconv(.c) TbOpType;
const FnIrTbopNumInputs = *const fn (?*const MirageTBGraph, usize) callconv(.c) usize;
const FnIrTbopNumOutputs = *const fn (?*const MirageTBGraph, usize) callconv(.c) usize;
const FnIrTbopInputSpec = *const fn (?*const MirageTBGraph, usize, usize, *STensorSpec) callconv(.c) MirageStatus;
const FnIrTbopOutputSpec = *const fn (?*const MirageTBGraph, usize, usize, *STensorSpec) callconv(.c) MirageStatus;
const FnIrTbopInputInfo = *const fn (?*const MirageTBGraph, usize, *TbInputInfo) callconv(.c) MirageStatus;
const FnIrTbopOutputInfo = *const fn (?*const MirageTBGraph, usize, *TbOutputInfo) callconv(.c) MirageStatus;

// ---------------------------------------------------------------------------
// Runtime-loaded symbols
// ---------------------------------------------------------------------------

var fn_status_string: ?FnStatusString = null;

var fn_graph_create: ?FnGraphCreate = null;
var fn_graph_destroy: ?FnGraphDestroy = null;
var fn_graph_new_input: ?FnGraphNewInput = null;
var fn_graph_matmul: ?FnGraphMatmul = null;
var fn_graph_unary: ?FnGraphUnary = null;
var fn_graph_binary: ?FnGraphBinary = null;
var fn_graph_reduction: ?FnGraphReduction = null;
var fn_graph_rms_norm: ?FnGraphRmsNorm = null;
var fn_graph_mark_output: ?FnGraphMarkOutput = null;

var fn_device_create: ?FnDeviceCreate = null;
var fn_device_destroy: ?FnDeviceDestroy = null;
var fn_device_mem_info: ?FnDeviceMemInfo = null;

var fn_search: ?FnSearch = null;
var fn_search_result_destroy: ?FnSearchResultDestroy = null;
var fn_search_result_count: ?FnSearchResultCount = null;
var fn_search_result_get: ?FnSearchResultGet = null;

var fn_transpile: ?FnTranspile = null;
var fn_source_destroy: ?FnSourceDestroy = null;
var fn_source_traits: ?FnSourceTraits = null;
var fn_source_code: ?FnSourceCode = null;
var fn_source_code_len: ?FnSourceCodeLen = null;
var fn_source_buf_size: ?FnSourceBufSize = null;
var fn_source_max_smem: ?FnSourceMaxSmem = null;
var fn_source_num_outputs: ?FnSourceNumOutputs = null;
var fn_source_output_spec: ?FnSourceOutputSpec = null;
var fn_source_num_kernels: ?FnSourceNumKernels = null;
var fn_source_kernel_meta: ?FnSourceKernelMeta = null;
var fn_source_kernel_num_args: ?FnSourceKernelNumArgs = null;
var fn_source_kernel_arg: ?FnSourceKernelArg = null;

var fn_ir_num_ops: ?FnIrNumOps = null;
var fn_ir_op_type: ?FnIrOpType = null;
var fn_ir_op_num_inputs: ?FnIrOpNumInputs = null;
var fn_ir_op_num_outputs: ?FnIrOpNumOutputs = null;
var fn_ir_op_input: ?FnIrOpInput = null;
var fn_ir_op_output: ?FnIrOpOutput = null;
var fn_ir_tensor_spec: ?FnIrTensorSpec = null;

var fn_ir_op_tbgraph: ?FnIrOpTbgraph = null;
var fn_ir_tbgraph_grid_dim: ?FnIrTbgraphGridDim = null;
var fn_ir_tbgraph_block_dim: ?FnIrTbgraphBlockDim = null;
var fn_ir_tbgraph_forloop_range: ?FnIrTbgraphForloopRange = null;
var fn_ir_tbgraph_reduction_dimx: ?FnIrTbgraphReductionDimx = null;
var fn_ir_tbgraph_num_ops: ?FnIrTbgraphNumOps = null;
var fn_ir_tbgraph_op_type: ?FnIrTbgraphOpType = null;
var fn_ir_tbop_num_inputs: ?FnIrTbopNumInputs = null;
var fn_ir_tbop_num_outputs: ?FnIrTbopNumOutputs = null;
var fn_ir_tbop_input_spec: ?FnIrTbopInputSpec = null;
var fn_ir_tbop_output_spec: ?FnIrTbopOutputSpec = null;
var fn_ir_tbop_input_info: ?FnIrTbopInputInfo = null;
var fn_ir_tbop_output_info: ?FnIrTbopOutputInfo = null;

var load_mutex: std.Thread.Mutex = .{};
var symbols_ready = false;

pub const LoadError = error{
    MirageSymbolMissing,
};

extern "c" fn dlsym(handle: *anyopaque, symbol: [*:0]const u8) ?*anyopaque;
extern "c" fn dlerror() ?[*:0]const u8;

pub fn ensure_loaded(handle: *anyopaque) LoadError!void {
    load_mutex.lock();
    defer load_mutex.unlock();

    if (symbols_ready) return;

    // types.h
    fn_status_string = try load_symbol(FnStatusString, handle, "mirage_status_string");

    // graph.h - graph building
    fn_graph_create = try load_symbol(FnGraphCreate, handle, "mirage_graph_create");
    fn_graph_destroy = try load_symbol(FnGraphDestroy, handle, "mirage_graph_destroy");
    fn_graph_new_input = try load_symbol(FnGraphNewInput, handle, "mirage_graph_new_input");
    fn_graph_matmul = try load_symbol(FnGraphMatmul, handle, "mirage_graph_matmul");
    fn_graph_unary = try load_symbol(FnGraphUnary, handle, "mirage_graph_unary");
    fn_graph_binary = try load_symbol(FnGraphBinary, handle, "mirage_graph_binary");
    fn_graph_reduction = load_symbol_optional(FnGraphReduction, handle, "mirage_graph_reduction");
    fn_graph_rms_norm = load_symbol_optional(FnGraphRmsNorm, handle, "mirage_graph_rms_norm");
    fn_graph_mark_output = try load_symbol(FnGraphMarkOutput, handle, "mirage_graph_mark_output");

    // graph.h - device
    fn_device_create = try load_symbol(FnDeviceCreate, handle, "mirage_device_create");
    fn_device_destroy = try load_symbol(FnDeviceDestroy, handle, "mirage_device_destroy");
    fn_device_mem_info = load_symbol_optional(FnDeviceMemInfo, handle, "mirage_device_mem_info");

    // graph.h - search
    fn_search = try load_symbol(FnSearch, handle, "mirage_search");
    fn_search_result_destroy = try load_symbol(FnSearchResultDestroy, handle, "mirage_search_result_destroy");
    fn_search_result_count = try load_symbol(FnSearchResultCount, handle, "mirage_search_result_count");
    fn_search_result_get = try load_symbol(FnSearchResultGet, handle, "mirage_search_result_get");

    // source.h
    fn_transpile = try load_symbol(FnTranspile, handle, "mirage_transpile");
    fn_source_destroy = try load_symbol(FnSourceDestroy, handle, "mirage_source_destroy");
    fn_source_traits = try load_symbol(FnSourceTraits, handle, "mirage_source_traits");
    fn_source_code = try load_symbol(FnSourceCode, handle, "mirage_source_code");
    fn_source_code_len = try load_symbol(FnSourceCodeLen, handle, "mirage_source_code_len");
    fn_source_buf_size = try load_symbol(FnSourceBufSize, handle, "mirage_source_buf_size");
    fn_source_max_smem = try load_symbol(FnSourceMaxSmem, handle, "mirage_source_max_smem");
    fn_source_num_outputs = try load_symbol(FnSourceNumOutputs, handle, "mirage_source_num_outputs");
    fn_source_output_spec = try load_symbol(FnSourceOutputSpec, handle, "mirage_source_output_spec");
    fn_source_num_kernels = try load_symbol(FnSourceNumKernels, handle, "mirage_source_num_kernels");
    fn_source_kernel_meta = try load_symbol(FnSourceKernelMeta, handle, "mirage_source_kernel_meta");
    fn_source_kernel_num_args = try load_symbol(FnSourceKernelNumArgs, handle, "mirage_source_kernel_num_args");
    fn_source_kernel_arg = try load_symbol(FnSourceKernelArg, handle, "mirage_source_kernel_arg");

    // ir.h - kernel graph walk
    fn_ir_num_ops = try load_symbol(FnIrNumOps, handle, "mirage_ir_num_ops");
    fn_ir_op_type = try load_symbol(FnIrOpType, handle, "mirage_ir_op_type");
    fn_ir_op_num_inputs = try load_symbol(FnIrOpNumInputs, handle, "mirage_ir_op_num_inputs");
    fn_ir_op_num_outputs = try load_symbol(FnIrOpNumOutputs, handle, "mirage_ir_op_num_outputs");
    fn_ir_op_input = try load_symbol(FnIrOpInput, handle, "mirage_ir_op_input");
    fn_ir_op_output = try load_symbol(FnIrOpOutput, handle, "mirage_ir_op_output");
    fn_ir_tensor_spec = try load_symbol(FnIrTensorSpec, handle, "mirage_ir_tensor_spec");

    // ir.h - threadblock graph
    fn_ir_op_tbgraph = try load_symbol(FnIrOpTbgraph, handle, "mirage_ir_op_tbgraph");
    fn_ir_tbgraph_grid_dim = try load_symbol(FnIrTbgraphGridDim, handle, "mirage_ir_tbgraph_grid_dim");
    fn_ir_tbgraph_block_dim = try load_symbol(FnIrTbgraphBlockDim, handle, "mirage_ir_tbgraph_block_dim");
    fn_ir_tbgraph_forloop_range = try load_symbol(FnIrTbgraphForloopRange, handle, "mirage_ir_tbgraph_forloop_range");
    fn_ir_tbgraph_reduction_dimx = try load_symbol(FnIrTbgraphReductionDimx, handle, "mirage_ir_tbgraph_reduction_dimx");
    fn_ir_tbgraph_num_ops = try load_symbol(FnIrTbgraphNumOps, handle, "mirage_ir_tbgraph_num_ops");
    fn_ir_tbgraph_op_type = try load_symbol(FnIrTbgraphOpType, handle, "mirage_ir_tbgraph_op_type");
    fn_ir_tbop_num_inputs = try load_symbol(FnIrTbopNumInputs, handle, "mirage_ir_tbop_num_inputs");
    fn_ir_tbop_num_outputs = try load_symbol(FnIrTbopNumOutputs, handle, "mirage_ir_tbop_num_outputs");
    fn_ir_tbop_input_spec = try load_symbol(FnIrTbopInputSpec, handle, "mirage_ir_tbop_input_spec");
    fn_ir_tbop_output_spec = try load_symbol(FnIrTbopOutputSpec, handle, "mirage_ir_tbop_output_spec");
    fn_ir_tbop_input_info = try load_symbol(FnIrTbopInputInfo, handle, "mirage_ir_tbop_input_info");
    fn_ir_tbop_output_info = try load_symbol(FnIrTbopOutputInfo, handle, "mirage_ir_tbop_output_info");

    symbols_ready = true;
}

// ---------------------------------------------------------------------------
// Public wrappers (thin forwarding)
// ---------------------------------------------------------------------------

pub fn mirage_status_string(status: MirageStatus) [*:0]const u8 {
    const f = fn_status_string orelse return "unknown_status";
    return f(status);
}

// graph.h - graph building

pub fn mirage_graph_create(out: *?*MirageGraph) MirageStatus {
    const f = fn_graph_create orelse return status_internal_error;
    return f(out);
}

pub fn mirage_graph_destroy(graph: ?*MirageGraph) void {
    const f = fn_graph_destroy orelse return;
    f(graph);
}

pub fn mirage_graph_new_input(graph: ?*MirageGraph, spec: *const TensorSpec, out: *MirageTensor) MirageStatus {
    const f = fn_graph_new_input orelse return status_internal_error;
    return f(graph, spec, out);
}

pub fn mirage_graph_matmul(graph: ?*MirageGraph, lhs: MirageTensor, rhs: MirageTensor, out: *MirageTensor) MirageStatus {
    const f = fn_graph_matmul orelse return status_internal_error;
    return f(graph, lhs, rhs, out);
}

pub fn mirage_graph_unary(graph: ?*MirageGraph, op: MirageUnaryOp, input: MirageTensor, out: *MirageTensor) MirageStatus {
    const f = fn_graph_unary orelse return status_internal_error;
    return f(graph, op, input, out);
}

pub fn mirage_graph_binary(graph: ?*MirageGraph, op: MirageBinaryOp, lhs: MirageTensor, rhs: MirageTensor, out: *MirageTensor) MirageStatus {
    const f = fn_graph_binary orelse return status_internal_error;
    return f(graph, op, lhs, rhs, out);
}

pub fn mirage_graph_reduction(graph: ?*MirageGraph, input: MirageTensor, dim: i32, factor: i32, out: *MirageTensor) MirageStatus {
    const f = fn_graph_reduction orelse return status_unsupported;
    return f(graph, input, dim, factor, out);
}

pub fn mirage_graph_rms_norm(graph: ?*MirageGraph, input: MirageTensor, normalized_size: i32, out: *MirageTensor) MirageStatus {
    const f = fn_graph_rms_norm orelse return status_unsupported;
    return f(graph, input, normalized_size, out);
}

pub fn mirage_graph_mark_output(graph: ?*MirageGraph, tensor: MirageTensor) MirageStatus {
    const f = fn_graph_mark_output orelse return status_internal_error;
    return f(graph, tensor);
}

// graph.h - device

pub fn mirage_device_create(ordinal: i32, out: *?*MirageDevice) MirageStatus {
    const f = fn_device_create orelse return status_internal_error;
    return f(ordinal, out);
}

pub fn mirage_device_destroy(device: ?*MirageDevice) void {
    const f = fn_device_destroy orelse return;
    f(device);
}

pub fn mirage_device_mem_info() ?struct { free: usize, total: usize } {
    const f = fn_device_mem_info orelse return null;
    var free: usize = 0;
    var total: usize = 0;
    const st = f(null, &free, &total);
    if (st != status_ok) return null;
    return .{ .free = free, .total = total };
}

// graph.h - search

pub fn mirage_search(device: ?*MirageDevice, graph: ?*const MirageGraph, options: ?*const SearchOptions, out: *?*MirageSearchResult) MirageStatus {
    const f = fn_search orelse return status_internal_error;
    return f(device, graph, options, out);
}

pub fn mirage_search_result_destroy(result: ?*MirageSearchResult) void {
    const f = fn_search_result_destroy orelse return;
    f(result);
}

pub fn mirage_search_result_count(result: ?*const MirageSearchResult) usize {
    const f = fn_search_result_count orelse return 0;
    return f(result);
}

pub fn mirage_search_result_get(result: ?*const MirageSearchResult, index: usize) ?*const MirageGraph {
    const f = fn_search_result_get orelse return null;
    return f(result, index);
}

// source.h

pub fn mirage_transpile(graph: ?*const MirageGraph, options: ?*const TranspileOptions, out: *?*MirageSource) MirageStatus {
    const f = fn_transpile orelse return status_internal_error;
    return f(graph, options, out);
}

pub fn mirage_source_destroy(source: ?*MirageSource) void {
    const f = fn_source_destroy orelse return;
    f(source);
}

pub fn mirage_source_traits(source: ?*const MirageSource) SourceTraits {
    const f = fn_source_traits orelse return 0;
    return f(source);
}

pub fn mirage_source_code(source: ?*const MirageSource) ?[*:0]const u8 {
    const f = fn_source_code orelse return null;
    return f(source);
}

pub fn mirage_source_code_len(source: ?*const MirageSource) usize {
    const f = fn_source_code_len orelse return 0;
    return f(source);
}

pub fn mirage_source_buf_size(source: ?*const MirageSource) usize {
    const f = fn_source_buf_size orelse return 0;
    return f(source);
}

pub fn mirage_source_max_smem(source: ?*const MirageSource) usize {
    const f = fn_source_max_smem orelse return 0;
    return f(source);
}

pub fn mirage_source_num_outputs(source: ?*const MirageSource) usize {
    const f = fn_source_num_outputs orelse return 0;
    return f(source);
}

pub fn mirage_source_output_spec(source: ?*const MirageSource, index: usize, out: *TensorSpec) MirageStatus {
    const f = fn_source_output_spec orelse return status_internal_error;
    return f(source, index, out);
}

pub fn mirage_source_num_kernels(source: ?*const MirageSource) usize {
    const f = fn_source_num_kernels orelse return 0;
    return f(source);
}

pub fn mirage_source_kernel_meta(source: ?*const MirageSource, index: usize, out: *KernelMeta) MirageStatus {
    const f = fn_source_kernel_meta orelse return status_internal_error;
    return f(source, index, out);
}

// source.h - kernel argument mapping

pub fn mirage_source_kernel_num_args(source: ?*const MirageSource, kernel_index: usize) usize {
    const f = fn_source_kernel_num_args orelse return 0;
    return f(source, kernel_index);
}

pub fn mirage_source_kernel_arg(source: ?*const MirageSource, kernel_index: usize, arg_index: usize, out: *KernelArg) MirageStatus {
    const f = fn_source_kernel_arg orelse return status_internal_error;
    return f(source, kernel_index, arg_index, out);
}

// ir.h - kernel graph walk

pub fn mirage_ir_num_ops(graph: ?*const MirageGraph) usize {
    const f = fn_ir_num_ops orelse return 0;
    return f(graph);
}

pub fn mirage_ir_op_type(graph: ?*const MirageGraph, op_index: usize) KnOpType {
    const f = fn_ir_op_type orelse return 0;
    return f(graph, op_index);
}

pub fn mirage_ir_op_num_inputs(graph: ?*const MirageGraph, op_index: usize) usize {
    const f = fn_ir_op_num_inputs orelse return 0;
    return f(graph, op_index);
}

pub fn mirage_ir_op_num_outputs(graph: ?*const MirageGraph, op_index: usize) usize {
    const f = fn_ir_op_num_outputs orelse return 0;
    return f(graph, op_index);
}

pub fn mirage_ir_op_input(graph: ?*const MirageGraph, op_index: usize, tensor_index: usize) MirageTensor {
    const f = fn_ir_op_input orelse return std.math.maxInt(MirageTensor);
    return f(graph, op_index, tensor_index);
}

pub fn mirage_ir_op_output(graph: ?*const MirageGraph, op_index: usize, tensor_index: usize) MirageTensor {
    const f = fn_ir_op_output orelse return std.math.maxInt(MirageTensor);
    return f(graph, op_index, tensor_index);
}

pub fn mirage_ir_tensor_spec(graph: ?*const MirageGraph, tensor: MirageTensor, out: *TensorSpec) MirageStatus {
    const f = fn_ir_tensor_spec orelse return status_internal_error;
    return f(graph, tensor, out);
}

// ir.h - threadblock graph

pub fn mirage_ir_op_tbgraph(graph: ?*const MirageGraph, op_index: usize, out: *?*const MirageTBGraph) MirageStatus {
    const f = fn_ir_op_tbgraph orelse return status_internal_error;
    return f(graph, op_index, out);
}

pub fn mirage_ir_tbgraph_grid_dim(tbg: ?*const MirageTBGraph, out: *[3]u32) void {
    const f = fn_ir_tbgraph_grid_dim orelse return;
    f(tbg, out);
}

pub fn mirage_ir_tbgraph_block_dim(tbg: ?*const MirageTBGraph, out: *[3]u32) void {
    const f = fn_ir_tbgraph_block_dim orelse return;
    f(tbg, out);
}

pub fn mirage_ir_tbgraph_forloop_range(tbg: ?*const MirageTBGraph) i32 {
    const f = fn_ir_tbgraph_forloop_range orelse return 0;
    return f(tbg);
}

pub fn mirage_ir_tbgraph_reduction_dimx(tbg: ?*const MirageTBGraph) i32 {
    const f = fn_ir_tbgraph_reduction_dimx orelse return 0;
    return f(tbg);
}

pub fn mirage_ir_tbgraph_num_ops(tbg: ?*const MirageTBGraph) usize {
    const f = fn_ir_tbgraph_num_ops orelse return 0;
    return f(tbg);
}

pub fn mirage_ir_tbgraph_op_type(tbg: ?*const MirageTBGraph, op_index: usize) TbOpType {
    const f = fn_ir_tbgraph_op_type orelse return 0;
    return f(tbg, op_index);
}

pub fn mirage_ir_tbop_num_inputs(tbg: ?*const MirageTBGraph, op_index: usize) usize {
    const f = fn_ir_tbop_num_inputs orelse return 0;
    return f(tbg, op_index);
}

pub fn mirage_ir_tbop_num_outputs(tbg: ?*const MirageTBGraph, op_index: usize) usize {
    const f = fn_ir_tbop_num_outputs orelse return 0;
    return f(tbg, op_index);
}

pub fn mirage_ir_tbop_input_spec(tbg: ?*const MirageTBGraph, op_index: usize, tensor_index: usize, out: *STensorSpec) MirageStatus {
    const f = fn_ir_tbop_input_spec orelse return status_internal_error;
    return f(tbg, op_index, tensor_index, out);
}

pub fn mirage_ir_tbop_output_spec(tbg: ?*const MirageTBGraph, op_index: usize, tensor_index: usize, out: *STensorSpec) MirageStatus {
    const f = fn_ir_tbop_output_spec orelse return status_internal_error;
    return f(tbg, op_index, tensor_index, out);
}

pub fn mirage_ir_tbop_input_info(tbg: ?*const MirageTBGraph, op_index: usize, out: *TbInputInfo) MirageStatus {
    const f = fn_ir_tbop_input_info orelse return status_internal_error;
    return f(tbg, op_index, out);
}

pub fn mirage_ir_tbop_output_info(tbg: ?*const MirageTBGraph, op_index: usize, out: *TbOutputInfo) MirageStatus {
    const f = fn_ir_tbop_output_info orelse return status_internal_error;
    return f(tbg, op_index, out);
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn load_symbol(comptime T: type, handle: *anyopaque, comptime symbol: [:0]const u8) LoadError!T {
    clear_dlerror();
    const raw = dlsym(handle, symbol.ptr) orelse {
        if (dlerror()) |err| {
            log.err("missing symbol {s}: {s}", .{ symbol, std.mem.span(err) });
        } else {
            log.err("missing symbol {s}", .{symbol});
        }
        return error.MirageSymbolMissing;
    };
    return @ptrCast(raw);
}

fn load_symbol_optional(comptime T: type, handle: *anyopaque, comptime symbol: [:0]const u8) ?T {
    clear_dlerror();
    const raw = dlsym(handle, symbol.ptr) orelse {
        _ = dlerror();
        return null;
    };
    return @ptrCast(raw);
}

fn clear_dlerror() void {
    _ = dlerror();
}
