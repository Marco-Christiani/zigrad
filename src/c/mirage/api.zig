//! ABI adapter for the Zigrad Mirage library.
//!
//! Optimization uses Mirage's symbolic search and returns one selected graph.
//!
//! Public declarations describe the Zigrad API. Translated declarations and
//!  upstream object layouts remain private to this adapter.
const std = @import("std");
const c = @import("c.zig");
const dylib = @import("../dylib.zig");

const log = std.log.scoped(.@"zg/mirage_api");

pub const MirageError = error{
    MirageUnavailable,
    MirageInvalidArgument,
    MirageInternalError,
    MirageApiUnsupported,
    MirageNotFound,
    OutOfMemory,
};

/// Graph-local tensor handle.
pub const Tensor = struct {
    index: u32,
};

pub const max_rank: usize = 4;

pub const DType = enum {
    f16,
    bf16,
    f32,
    f64,
};

pub const UnaryOp = enum {
    exp,
    sqrt,
    silu,
    gelu,
    relu,
    log,
};

pub const BinaryOp = enum {
    add,
    mul,
    div,
    pow,
};

pub const SearchPreset = enum {
    default,
    attention,
    lora,
    mlp,
};

pub const TensorSpec = struct {
    dtype: DType,
    dims: []const i64,
    /// Element strides. Null requests a dense row-major layout.
    strides: ?[]const i64 = null,
};

pub const OptimizeOptions = struct {
    /// Search limit in seconds. A non-positive value uses Mirage's default.
    time_limit_seconds: f64 = 0,

    /// Optional checkpoint path. Null disables checkpoint I/O.
    checkpoint_path: ?[:0]const u8 = null,

    /// Named upstream search configuration.
    preset: SearchPreset = .default,

    /// Emit upstream search diagnostics.
    verbose: bool = false,
};

pub const TranspileOptions = struct {
    /// CUDA compute capability times ten. Zero uses the session device.
    target_cc: i32 = 0,

    /// Pipeline depth. Zero uses the adapter default.
    pipeline_stages: i32 = 0,
};

pub const KernelMeta = struct {
    /// Function name freed by `Source.deinit`.
    func_name: []const u8,
    smem_bytes: usize,
    grid_dim: [3]u32,
    block_dim: [3]u32,
    args: []const KernelArg,
};

pub const ArgSource = enum {
    input,
    output,
    workspace,
};

pub const KernelArg = struct {
    source: ArgSource,
    index_or_offset: usize,
};

pub const Source = struct {
    allocator: std.mem.Allocator,
    code: []const u8,
    workspace_size: usize,
    kernels: []const KernelMeta,

    pub fn deinit(self: *Source) void {
        for (self.kernels) |kernel| {
            self.allocator.free(kernel.func_name);
            self.allocator.free(kernel.args);
        }
        self.allocator.free(self.kernels);
        self.allocator.free(self.code);
        self.* = undefined;
    }
};

const RuntimeState = union(enum) {
    rejected: dylib.Library,
    ready: dylib.Library,
};

var runtime_state: ?RuntimeState = null;
var runtime_load_mutex: std.Io.Mutex = .init;

/// Load the Zigrad Mirage adapter from one explicit path.
///
/// The first successful load retains the adapter handle for the process lifetime.
pub fn load_runtime(path: []const u8) MirageError!void {
    std.Io.Threaded.mutexLock(&runtime_load_mutex);
    defer std.Io.Threaded.mutexUnlock(&runtime_load_mutex);

    if (runtime_state) |state| {
        return switch (state) {
            .ready => {},
            .rejected => error.MirageUnavailable,
        };
    }

    var library = dylib.Library.open(std.heap.smp_allocator, path, .{
        .visibility = .global,
    }) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        error.OpenFailed => {
            log.err("failed to open Mirage adapter '{s}': {s}", .{ path, dylib.error_message() });
            return error.MirageUnavailable;
        },
    };

    c.install_symbols(library) catch {
        library.close();
        return error.MirageUnavailable;
    };
    runtime_state = .{ .rejected = library };

    const actual_abi = c.mirage_abi_version();
    if (actual_abi != c.abi_version) {
        log.err("Mirage adapter ABI mismatch: expected {d}, found {d}", .{ c.abi_version, actual_abi });
        return error.MirageUnavailable;
    }

    const capabilities = c.mirage_capabilities();
    if (capabilities & c.required_capabilities != c.required_capabilities) {
        log.err("Mirage adapter lacks required capabilities: found 0x{x}", .{capabilities});
        return error.MirageUnavailable;
    }

    runtime_state = .{ .ready = library };
    log.info("loaded Mirage adapter '{s}'", .{path});
}

fn require_loaded() MirageError!void {
    const state = runtime_state orelse return error.MirageUnavailable;
    switch (state) {
        .ready => {},
        .rejected => return error.MirageUnavailable,
    }
}

fn check_status(status: c.MirageStatus) MirageError!void {
    if (status == c.status_ok) return;

    const detail = std.mem.span(c.mirage_last_error());
    if (detail.len != 0) {
        log.debug("Mirage call failed ({s}): {s}", .{
            std.mem.span(c.mirage_status_string(status)),
            detail,
        });
    }

    if (status == c.status_invalid_argument) return error.MirageInvalidArgument;
    if (status == c.status_unsupported) return error.MirageApiUnsupported;
    if (status == c.status_not_found) return error.MirageNotFound;
    if (status == c.status_out_of_memory) return error.OutOfMemory;
    return error.MirageInternalError;
}

pub const Device = opaque {
    pub fn init(ordinal: i32) MirageError!*Device {
        try require_loaded();
        var raw: ?*c.MirageSession = null;
        try check_status(c.mirage_session_create(ordinal, &raw));
        return @ptrCast(raw orelse return error.MirageInternalError);
    }

    pub fn deinit(self: *Device) void {
        c.mirage_session_destroy(@ptrCast(self));
    }
};

pub const Graph = opaque {
    pub fn init() MirageError!*Graph {
        try require_loaded();
        var raw: ?*c.MirageGraph = null;
        try check_status(c.mirage_graph_create(&raw));
        return @ptrCast(raw orelse return error.MirageInternalError);
    }

    pub fn deinit(self: *Graph) void {
        c.mirage_graph_destroy(@ptrCast(self));
    }

    pub fn new_input(self: *Graph, spec: *const TensorSpec) MirageError!Tensor {
        if (spec.dims.len == 0 or spec.dims.len > max_rank) {
            return error.MirageInvalidArgument;
        }
        if (spec.strides) |strides| {
            if (strides.len != spec.dims.len) {
                return error.MirageInvalidArgument;
            }
        }

        var raw_spec: c.TensorSpec = .{
            .dtype = dtype_to_c(spec.dtype),
            .rank = @intCast(spec.dims.len),
            .dims = @splat(0),
            .strides = @splat(0),
        };
        @memcpy(raw_spec.dims[0..spec.dims.len], spec.dims);
        if (spec.strides) |strides| {
            @memcpy(raw_spec.strides[0..strides.len], strides);
        }

        var raw_tensor: c.MirageTensor = 0;
        try check_status(c.mirage_graph_new_input(@ptrCast(self), &raw_spec, &raw_tensor));
        return .{ .index = raw_tensor };
    }

    pub fn matmul(self: *Graph, lhs: Tensor, rhs: Tensor) MirageError!Tensor {
        var raw_out: c.MirageTensor = 0;
        try check_status(c.mirage_graph_matmul(@ptrCast(self), lhs.index, rhs.index, &raw_out));
        return .{ .index = raw_out };
    }

    pub fn unary(self: *Graph, op: UnaryOp, input: Tensor) MirageError!Tensor {
        var raw_out: c.MirageTensor = 0;
        try check_status(c.mirage_graph_unary(@ptrCast(self), unary_op_to_c(op), input.index, &raw_out));
        return .{ .index = raw_out };
    }

    pub fn binary(self: *Graph, op: BinaryOp, lhs: Tensor, rhs: Tensor) MirageError!Tensor {
        var raw_out: c.MirageTensor = 0;
        try check_status(c.mirage_graph_binary(@ptrCast(self), binary_op_to_c(op), lhs.index, rhs.index, &raw_out));
        return .{ .index = raw_out };
    }

    pub fn reduction(self: *Graph, input: Tensor, dim: i32, factor: i32) MirageError!Tensor {
        var raw_out: c.MirageTensor = 0;
        try check_status(c.mirage_graph_reduction(@ptrCast(self), input.index, dim, factor, &raw_out));
        return .{ .index = raw_out };
    }

    pub fn rms_norm(self: *Graph, input: Tensor, normalized_size: i32) MirageError!Tensor {
        var raw_out: c.MirageTensor = 0;
        try check_status(c.mirage_graph_rms_norm(@ptrCast(self), input.index, normalized_size, &raw_out));
        return .{ .index = raw_out };
    }

    pub fn mark_output(self: *Graph, tensor: Tensor) MirageError!void {
        try check_status(c.mirage_graph_mark_output(@ptrCast(self), tensor.index));
    }
};

pub const OptimizedGraph = opaque {
    pub fn deinit(self: *OptimizedGraph) void {
        c.mirage_graph_destroy(@ptrCast(self));
    }
};

/// Runs Mirage symbolic optimization and returns one selected graph.
pub fn optimize(
    device: *Device,
    graph: *const Graph,
    options: ?*const OptimizeOptions,
) MirageError!*OptimizedGraph {
    try require_loaded();

    var raw_options: c.OptimizeOptions = undefined;
    const raw_options_ptr: ?*const c.OptimizeOptions = if (options) |opts| options: {
        raw_options = .{
            .struct_size = @sizeOf(c.OptimizeOptions),
            .time_limit_seconds = opts.time_limit_seconds,
            .checkpoint_path = if (opts.checkpoint_path) |path| path.ptr else null,
            .preset = search_preset_to_c(opts.preset),
            .verbose = @intFromBool(opts.verbose),
        };
        break :options &raw_options;
    } else null;

    var raw: ?*c.MirageGraph = null;
    try check_status(c.mirage_optimize(
        @ptrCast(device),
        @ptrCast(graph),
        raw_options_ptr,
        &raw,
    ));
    return @ptrCast(raw orelse return error.MirageInternalError);
}

/// Transpiles an optimized graph after probing it in a child process.
///
/// `Source.deinit` frees the result and nested data with `allocator`.
pub fn transpile(
    allocator: std.mem.Allocator,
    graph: *const OptimizedGraph,
    options: ?*const TranspileOptions,
) MirageError!Source {
    try require_loaded();
    if (!probe_transpile_safe(graph, options)) {
        return error.MirageApiUnsupported;
    }
    const raw = try transpile_loaded(graph, options);
    defer c.mirage_source_destroy(raw);
    return try copy_source(allocator, raw);
}

fn transpile_loaded(
    graph: *const OptimizedGraph,
    options: ?*const TranspileOptions,
) MirageError!*c.MirageSource {
    var raw_options: c.TranspileOptions = undefined;
    const raw_options_ptr: ?*const c.TranspileOptions = if (options) |opts| options: {
        raw_options = .{
            .struct_size = @sizeOf(c.TranspileOptions),
            .target_cc = opts.target_cc,
            .pipeline_stages = opts.pipeline_stages,
        };
        break :options &raw_options;
    } else null;

    var raw: ?*c.MirageSource = null;
    try check_status(c.mirage_transpile(@ptrCast(graph), raw_options_ptr, &raw));
    return raw orelse return error.MirageInternalError;
}

fn copy_source(
    allocator: std.mem.Allocator,
    raw_source: *const c.MirageSource,
) MirageError!Source {
    const raw_code = c.mirage_source_code(raw_source);
    const code = try allocator.dupe(
        u8,
        raw_code[0..c.mirage_source_code_len(raw_source)],
    );
    errdefer allocator.free(code);

    const num_kernels = c.mirage_source_num_kernels(raw_source);
    const kernels = try allocator.alloc(KernelMeta, num_kernels);
    errdefer allocator.free(kernels);

    var initialized: usize = 0;
    errdefer {
        for (kernels[0..initialized]) |kernel| {
            allocator.free(kernel.func_name);
            allocator.free(kernel.args);
        }
    }

    for (kernels, 0..) |*kernel, kernel_index| {
        var raw_meta: c.RawKernelMeta = std.mem.zeroes(c.RawKernelMeta);
        try check_status(c.mirage_source_kernel_meta(
            raw_source,
            kernel_index,
            &raw_meta,
        ));

        const raw_name = if (raw_meta.function_name) |ptr|
            ptr[0..raw_meta.function_name_len]
        else if (raw_meta.function_name_len == 0)
            ""
        else
            return error.MirageInternalError;
        const func_name = try allocator.dupe(u8, raw_name);
        errdefer allocator.free(func_name);

        const num_args = c.mirage_source_kernel_num_args(raw_source, kernel_index);
        const args = try allocator.alloc(KernelArg, num_args);
        errdefer allocator.free(args);
        for (args, 0..) |*arg, arg_index| {
            var raw_arg: c.RawKernelArg = std.mem.zeroes(c.RawKernelArg);
            try check_status(c.mirage_source_kernel_arg(
                raw_source,
                kernel_index,
                arg_index,
                &raw_arg,
            ));
            arg.* = .{
                .source = try arg_source_from_c(raw_arg.source),
                .index_or_offset = raw_arg.index_or_offset,
            };
        }

        var grid_dim: [3]u32 = undefined;
        var block_dim: [3]u32 = undefined;
        for (0..3) |dim| {
            grid_dim[dim] = @intCast(raw_meta.grid_dim[dim]);
            block_dim[dim] = @intCast(raw_meta.block_dim[dim]);
        }

        kernel.* = .{
            .func_name = func_name,
            .smem_bytes = raw_meta.shared_memory_bytes,
            .grid_dim = grid_dim,
            .block_dim = block_dim,
            .args = args,
        };
        initialized += 1;
    }

    return .{
        .allocator = allocator,
        .code = code,
        .workspace_size = c.mirage_source_workspace_size(raw_source),
        .kernels = kernels,
    };
}

fn probe_transpile_safe(
    graph: *const OptimizedGraph,
    options: ?*const TranspileOptions,
) bool {
    const fork_result = std.posix.system.fork();
    switch (std.posix.errno(fork_result)) {
        .SUCCESS => {},
        else => |err| {
            log.warn("fork failed for transpile probe: {s}", .{@tagName(err)});
            return false;
        },
    }
    const pid: std.posix.pid_t = @intCast(fork_result);

    if (pid == 0) {
        if (std.posix.openat(std.posix.AT.FDCWD, "/dev/null", .{ .ACCMODE = .WRONLY }, 0)) |devnull| {
            std.Io.Threaded.dup2(devnull, std.posix.STDERR_FILENO) catch {};
            std.Io.Threaded.closeFd(devnull);
        } else |_| {}

        const raw_source = transpile_loaded(graph, options) catch {
            std.os.linux.exit_group(1);
        };
        c.mirage_source_destroy(raw_source);
        std.os.linux.exit_group(0);
    }

    var status: c_int = undefined;
    while (true) {
        switch (std.posix.errno(std.posix.system.waitpid(pid, &status, 0))) {
            .SUCCESS => break,
            .INTR => continue,
            else => |err| {
                log.warn("waitpid failed for transpile probe: {s}", .{@tagName(err)});
                return false;
            },
        }
    }

    const status_bits: u32 = @bitCast(status);
    const W = std.os.linux.W;
    return W.IFEXITED(status_bits) and W.EXITSTATUS(status_bits) == 0;
}

fn dtype_to_c(dtype: DType) c_uint {
    return switch (dtype) {
        .f16 => c.dtype_f16,
        .bf16 => c.dtype_bf16,
        .f32 => c.dtype_f32,
        .f64 => c.dtype_f64,
    };
}

fn unary_op_to_c(op: UnaryOp) c_uint {
    return switch (op) {
        .exp => c.unary_exp,
        .sqrt => c.unary_sqrt,
        .silu => c.unary_silu,
        .gelu => c.unary_gelu,
        .relu => c.unary_relu,
        .log => c.unary_log,
    };
}

fn binary_op_to_c(op: BinaryOp) c_uint {
    return switch (op) {
        .add => c.binary_add,
        .mul => c.binary_mul,
        .div => c.binary_div,
        .pow => c.binary_pow,
    };
}

fn search_preset_to_c(preset: SearchPreset) c_uint {
    return switch (preset) {
        .default => c.search_default,
        .attention => c.search_attention,
        .lora => c.search_lora,
        .mlp => c.search_mlp,
    };
}

fn arg_source_from_c(source: c_uint) MirageError!ArgSource {
    if (source == c.arg_input) return .input;
    if (source == c.arg_output) return .output;
    if (source == c.arg_workspace) return .workspace;
    return error.MirageInternalError;
}

test "public handles are opaque Zig types" {
    inline for (.{ Device, Graph, OptimizedGraph }) |Handle| {
        switch (@typeInfo(Handle)) {
            .@"opaque" => {},
            else => return error.TestUnexpectedResult,
        }
    }
}

test "public value types do not alias translated C declarations" {
    try std.testing.expect(Tensor != c.MirageTensor);
    try std.testing.expect(TensorSpec != c.TensorSpec);
    try std.testing.expect(DType != @TypeOf(c.dtype_f32));
    try std.testing.expect(KernelArg != c.RawKernelArg);
    try std.testing.expect(Source != c.MirageSource);
}
