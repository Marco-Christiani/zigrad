const std = @import("std");
const c = @import("c.zig");

const log = std.log.scoped(.@"zg/mirage_api");

pub const MirageError = error{
    MirageUnavailable,
    MirageInvalidArgument,
    MirageInternalError,
    MirageApiUnsupported,
    MirageNotFound,
    OutOfMemory,
};

const RTLD_NOW: c_int = 0x2;
const RTLD_GLOBAL: c_int = 0x100;
extern "c" fn dlopen(filename: [*:0]const u8, flags: c_int) ?*anyopaque;
extern "c" fn dlerror() ?[*:0]const u8;

var runtime_lib_handle: ?*anyopaque = null;
var runtime_load_mutex: std.Thread.Mutex = .{};

pub fn ensure_loaded() MirageError!void {
    runtime_load_mutex.lock();
    defer runtime_load_mutex.unlock();

    if (runtime_lib_handle == null) {
        runtime_lib_handle = try load_runtime_library();
    }

    c.ensure_loaded(runtime_lib_handle.?) catch {
        return error.MirageUnavailable;
    };
}

fn load_runtime_library() MirageError!*anyopaque {
    if (dlopen("libmirage_runtime.so", RTLD_NOW | RTLD_GLOBAL)) |h| {
        return h;
    } else if (dlerror()) |err| {
        log.debug("dlopen(libmirage_runtime.so) failed: {s}", .{std.mem.span(err)});
    }

    const allocator = std.heap.smp_allocator;

    var has_static_archive = false;

    const runtime_handle = try load_runtime_from_sdk_root(allocator, "ZG_RUNTIME_SDK_ROOT", &has_static_archive);
    if (runtime_handle) |h| return h;

    const external_handle = try load_runtime_from_sdk_root(allocator, "ZG_EXTERNAL_SDK_ROOT", &has_static_archive);
    if (external_handle) |h| return h;

    if (has_static_archive) {
        log.err("mirage runtime shared library missing; found static archive only (libmirage_runtime.a)", .{});
    } else {
        log.err("mirage runtime shared library not found (libmirage_runtime.so)", .{});
    }
    return error.MirageUnavailable;
}

fn load_runtime_from_sdk_root(
    allocator: std.mem.Allocator,
    comptime env_name: []const u8,
    has_static_archive: *bool,
) MirageError!?*anyopaque {
    const sdk_root = std.process.getEnvVarOwned(allocator, env_name) catch |err| switch (err) {
        error.EnvironmentVariableNotFound => return null,
        error.OutOfMemory => return error.OutOfMemory,
        else => return error.MirageUnavailable,
    };
    defer allocator.free(sdk_root);

    const so_path = try std.fmt.allocPrint(allocator, "{s}/lib/libmirage_runtime.so", .{sdk_root});
    defer allocator.free(so_path);

    const so_path_z = try allocator.allocSentinel(u8, so_path.len, 0);
    defer allocator.free(so_path_z);
    @memcpy(so_path_z[0..so_path.len], so_path);

    if (dlopen(so_path_z.ptr, RTLD_NOW | RTLD_GLOBAL)) |h| {
        return h;
    } else if (dlerror()) |err| {
        log.debug("dlopen({s}) failed: {s}", .{ so_path, std.mem.span(err) });
    }

    const a_path = try std.fmt.allocPrint(allocator, "{s}/lib/libmirage_runtime.a", .{sdk_root});
    defer allocator.free(a_path);
    std.fs.accessAbsolute(a_path, .{}) catch return null;
    has_static_archive.* = true;
    return null;
}

pub fn status_name(status: c.MirageStatus) []const u8 {
    return std.mem.span(c.mirage_status_string(status));
}

fn check_status(status: c.MirageStatus) MirageError!void {
    if (status == c.status_ok) return;
    if (status == c.status_invalid_argument) return error.MirageInvalidArgument;
    if (status == c.status_internal_error) return error.MirageInternalError;
    if (status == c.status_unsupported) return error.MirageApiUnsupported;
    if (status == c.status_not_found) return error.MirageNotFound;
    return error.MirageInternalError;
}

// ---------------------------------------------------------------------------
// Device
// ---------------------------------------------------------------------------

pub const Device = struct {
    raw: ?*c.MirageDevice,

    pub fn init(ordinal: i32) MirageError!Device {
        try ensure_loaded();
        var raw: ?*c.MirageDevice = null;
        try check_status(c.mirage_device_create(ordinal, &raw));
        return .{ .raw = raw };
    }

    pub fn deinit(self: *Device) void {
        c.mirage_device_destroy(self.raw);
        self.raw = null;
    }

    pub fn memInfo(self: Device) ?struct { free: usize, total: usize } {
        _ = self;
        return c.mirage_device_mem_info();
    }
};

// ---------------------------------------------------------------------------
// Graph
// ---------------------------------------------------------------------------

pub const Graph = struct {
    raw: ?*c.MirageGraph,

    pub fn init() MirageError!Graph {
        try ensure_loaded();
        var raw: ?*c.MirageGraph = null;
        try check_status(c.mirage_graph_create(&raw));
        return .{ .raw = raw };
    }

    pub fn deinit(self: *Graph) void {
        c.mirage_graph_destroy(self.raw);
        self.raw = null;
    }

    pub fn newInput(self: *Graph, spec: *const c.TensorSpec) MirageError!c.MirageTensor {
        var tensor: c.MirageTensor = 0;
        try check_status(c.mirage_graph_new_input(self.raw, spec, &tensor));
        return tensor;
    }

    pub fn matmul(self: *Graph, lhs: c.MirageTensor, rhs: c.MirageTensor) MirageError!c.MirageTensor {
        var out: c.MirageTensor = 0;
        try check_status(c.mirage_graph_matmul(self.raw, lhs, rhs, &out));
        return out;
    }

    pub fn unary(self: *Graph, op: c.MirageUnaryOp, input: c.MirageTensor) MirageError!c.MirageTensor {
        var out: c.MirageTensor = 0;
        try check_status(c.mirage_graph_unary(self.raw, op, input, &out));
        return out;
    }

    pub fn binary(self: *Graph, op: c.MirageBinaryOp, lhs: c.MirageTensor, rhs: c.MirageTensor) MirageError!c.MirageTensor {
        var out: c.MirageTensor = 0;
        try check_status(c.mirage_graph_binary(self.raw, op, lhs, rhs, &out));
        return out;
    }

    pub fn reduction(self: *Graph, input: c.MirageTensor, dim: i32, factor: i32) MirageError!c.MirageTensor {
        var out: c.MirageTensor = 0;
        try check_status(c.mirage_graph_reduction(self.raw, input, dim, factor, &out));
        return out;
    }

    pub fn rmsNorm(self: *Graph, input: c.MirageTensor, normalized_size: i32) MirageError!c.MirageTensor {
        var out: c.MirageTensor = 0;
        try check_status(c.mirage_graph_rms_norm(self.raw, input, normalized_size, &out));
        return out;
    }

    pub fn markOutput(self: *Graph, tensor: c.MirageTensor) MirageError!void {
        try check_status(c.mirage_graph_mark_output(self.raw, tensor));
    }
};

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

pub const SearchResult = struct {
    raw: ?*c.MirageSearchResult,

    pub fn deinit(self: *SearchResult) void {
        c.mirage_search_result_destroy(self.raw);
        self.raw = null;
    }

    pub fn count(self: SearchResult) usize {
        return c.mirage_search_result_count(self.raw);
    }

    pub fn get(self: SearchResult, index: usize) ?*const c.MirageGraph {
        return c.mirage_search_result_get(self.raw, index);
    }
};

pub fn search(device: *Device, graph: *const Graph, options: ?*const c.SearchOptions) MirageError!SearchResult {
    try ensure_loaded();
    var raw: ?*c.MirageSearchResult = null;
    try check_status(c.mirage_search(device.raw, graph.raw, options, &raw));
    return .{ .raw = raw };
}

// ---------------------------------------------------------------------------
// Source (transpiled CUDA)
// ---------------------------------------------------------------------------

pub const Source = struct {
    raw: ?*c.MirageSource,

    pub fn deinit(self: *Source) void {
        c.mirage_source_destroy(self.raw);
        self.raw = null;
    }

    pub fn traits(self: Source) c.SourceTraits {
        return c.mirage_source_traits(self.raw);
    }

    pub fn code(self: Source) []const u8 {
        const ptr = c.mirage_source_code(self.raw) orelse return "";
        const len = c.mirage_source_code_len(self.raw);
        return ptr[0..len];
    }

    pub fn bufSize(self: Source) usize {
        return c.mirage_source_buf_size(self.raw);
    }

    pub fn maxSmem(self: Source) usize {
        return c.mirage_source_max_smem(self.raw);
    }

    pub fn numOutputs(self: Source) usize {
        return c.mirage_source_num_outputs(self.raw);
    }

    pub fn outputSpec(self: Source, index: usize) MirageError!c.TensorSpec {
        var spec: c.TensorSpec = std.mem.zeroes(c.TensorSpec);
        try check_status(c.mirage_source_output_spec(self.raw, index, &spec));
        return spec;
    }

    pub fn numKernels(self: Source) usize {
        return c.mirage_source_num_kernels(self.raw);
    }

    pub fn kernelMeta(self: Source, index: usize) MirageError!c.KernelMeta {
        var meta: c.KernelMeta = std.mem.zeroes(c.KernelMeta);
        try check_status(c.mirage_source_kernel_meta(self.raw, index, &meta));
        return meta;
    }
};

pub fn transpile(graph: ?*const c.MirageGraph, options: ?*const c.TranspileOptions) MirageError!Source {
    try ensure_loaded();
    var raw: ?*c.MirageSource = null;
    try check_status(c.mirage_transpile(graph, options, &raw));
    return .{ .raw = raw };
}
