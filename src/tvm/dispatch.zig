//! TVM kernel dispatch state.
//!
//! Owns a cache of loaded TVM runtime modules and provides a dispatch
//!  function conforming to `kernel.DispatchFn`.
const std = @import("std");
const kernel = @import("../kernel.zig");
const dlpack = @import("../c/dlpack.zig");
const tvm_api = @import("../c/tvm/api.zig");
const tvm_runtime = @import("../c/tvm/runtime.zig");
const tvm_c = @import("../c/tvm/c.zig");
const Cache = @import("../cache.zig").Cache;

const log = std.log.scoped(.@"zg/tvm_dispatch");

const TvmDispatchEntry = struct {
    module: tvm_runtime.RuntimeModule,
    main_func: tvm_api.Value,
};

/// Encapsulates TVM dispatch state: a thread-safe cache of loaded
/// TVM runtime modules keyed by kernel name.
///
/// A single instance is shared across all TVM kernel artifacts and
/// passed to them as the `dispatch_ctx` pointer. The `dispatch` method
/// conforms to `kernel.DispatchFn`.
pub const TvmDispatchState = struct {
    cache_mutex: std.Thread.Mutex = .{},
    cache: std.StringHashMap(TvmDispatchEntry),
    allocator: std.mem.Allocator,
    artifact_cache: Cache,

    pub fn init(allocator: std.mem.Allocator, artifact_cache: Cache) TvmDispatchState {
        return .{
            .cache = std.StringHashMap(TvmDispatchEntry).init(allocator),
            .allocator = allocator,
            .artifact_cache = artifact_cache,
        };
    }

    pub fn deinit(self: *TvmDispatchState) void {
        var it = self.cache.iterator();
        while (it.next()) |entry| {
            var module = entry.value_ptr.module;
            module.deinit();
            self.allocator.free(entry.key_ptr.*);
        }
        self.cache.deinit();
    }

    /// Provider dispatch entry point. Conforms to `kernel.DispatchFn`.
    pub fn dispatch(
        provider_ctx: *anyopaque,
        artifact_data: []const u8,
        kernel_key: []const u8,
        ctx: kernel.DispatchContext,
    ) kernel.DispatchError!void {
        const self: *TvmDispatchState = @ptrCast(@alignCast(provider_ctx));
        self.dispatch_impl(artifact_data, kernel_key, ctx) catch |err| {
            log.err("tvm dispatch failed for '{s}': {s}", .{ kernel_key, @errorName(err) });
            return error.DispatchFailed;
        };
    }

    fn dispatch_impl(
        self: *TvmDispatchState,
        artifact_data: []const u8,
        kernel_key: []const u8,
        ctx: kernel.DispatchContext,
    ) !void {
        try tvm_api.ensure_loaded(std.heap.c_allocator, .{});

        var entry: TvmDispatchEntry = undefined;
        {
            self.cache_mutex.lock();
            defer self.cache_mutex.unlock();

            if (self.cache.get(kernel_key)) |cached| {
                entry = cached;
            } else {
                const loaded = try load_dispatch_entry(self.artifact_cache, kernel_key, artifact_data);
                const cache_key = try self.allocator.dupe(u8, kernel_key);
                try self.cache.put(cache_key, loaded);
                entry = loaded;
            }
        }

        if (ctx.platform == .cuda) {
            if (ctx.stream) |stream_ptr| {
                try configure_cuda_stream(stream_ptr, ctx.device_ordinal);
            }
        }

        const device_type: dlpack.DeviceType = if (ctx.platform == .cuda) .cuda else .cpu;

        // Build DLPack tensors from BufferDescs and call the TVM function.
        var tvm_args: [16]tvm_api.Value = undefined;
        var tensors: [16]tvm_runtime.Tensor = undefined;
        const total = ctx.inputs.len + ctx.outputs.len;
        if (total > 16) return error.TvmCallFailed;

        for (ctx.inputs, 0..) |buf, i| {
            tensors[i] = try tensor_from_buffer_desc(buf, device_type, ctx.device_ordinal);
            tvm_args[i] = tensors[i].as_value();
        }
        for (ctx.outputs, 0..) |buf, i| {
            const idx = ctx.inputs.len + i;
            tensors[idx] = try tensor_from_buffer_desc(buf, device_type, ctx.device_ordinal);
            tvm_args[idx] = tensors[idx].as_value();
        }
        defer for (0..total) |i| tensors[i].deinit();

        const func_handle = entry.main_func.as_object() orelse return error.TvmCallFailed;
        _ = try tvm_api.call_handle(std.heap.c_allocator, func_handle, tvm_args[0..total]);
    }
};

// ============================================================================
// Internal helpers
// ============================================================================

fn tensor_from_buffer_desc(buf: kernel.BufferDesc, device_type: dlpack.DeviceType, device_id: i32) !tvm_runtime.Tensor {
    const dl_tensor: dlpack.Tensor = .{
        .data = buf.data,
        .device = .{ .device_type = device_type, .device_id = device_id },
        .ndim = @intCast(buf.rank),
        .dtype = kernel_dtype_to_dlpack(buf.dtype),
        .shape = @constCast(buf.dims.ptr),
        .strides = null,
        .byte_offset = 0,
    };
    const managed = try dlpack.ManagedTensor.heap_borrowing(std.heap.c_allocator, dl_tensor);
    return tvm_runtime.Tensor.from_dlpack(managed);
}

fn kernel_dtype_to_dlpack(dtype: kernel.DType) dlpack.DataType {
    return switch (dtype) {
        .f16 => .{ .code = .float, .bits = 16, .lanes = 1 },
        .bf16 => .{ .code = .bfloat, .bits = 16, .lanes = 1 },
        .f32 => .{ .code = .float, .bits = 32, .lanes = 1 },
        .f64 => .{ .code = .float, .bits = 64, .lanes = 1 },
        .i8 => .{ .code = .int, .bits = 8, .lanes = 1 },
        .u8 => .{ .code = .uint, .bits = 8, .lanes = 1 },
        .i32 => .{ .code = .int, .bits = 32, .lanes = 1 },
        .i64 => .{ .code = .int, .bits = 64, .lanes = 1 },
        .u32 => .{ .code = .uint, .bits = 32, .lanes = 1 },
        .u64 => .{ .code = .uint, .bits = 64, .lanes = 1 },
        .bool => .{ .code = .uint, .bits = 8, .lanes = 1 },
    };
}

fn configure_cuda_stream(stream_ptr: *anyopaque, device_id: i32) !void {
    _ = try tvm_api.call_global(std.heap.c_allocator, "runtime.TVMSetStream", &.{
        tvm_api.Value.int(2), // kDLCUDA
        tvm_api.Value.int(device_id),
        opaque_ptr_value(stream_ptr),
    });
}

fn opaque_ptr_value(ptr: *anyopaque) tvm_api.Value {
    var v = std.mem.zeroes(tvm_c.TVMFFIAny);
    v.type_index = tvm_c.kTVMFFIOpaquePtr;
    v.unnamed_1.v_int64 = @bitCast(@intFromPtr(ptr));
    return .{ .raw = v };
}

/// TODO: artifact materialization belongs in the provider, not dispatch.
///  TVM's C API requires a file path (load_from_file), so we write the .so
///  bytes to disk here as a trampoline. This should move to TvmProvider
///  during store-population (after tuning), so dispatch receives a path or
///  pre-loaded handle rather than raw bytes it must write out.
fn load_dispatch_entry(artifact_cache: Cache, kernel_key: []const u8, artifact_data: []const u8) !TvmDispatchEntry {
    const hash = std.hash.Wyhash.hash(0, kernel_key);
    var name_buf: [128]u8 = undefined;
    const filename = try std.fmt.bufPrint(&name_buf, "{x}.so", .{hash});
    const dispatch_cache = try artifact_cache.subdir("tvm/dispatch", .{});
    var resolved = try dispatch_cache.join(filename);
    const path = resolved.pathZ();

    const file = try std.fs.cwd().createFile(path, .{ .truncate = true });
    defer file.close();
    try file.writeAll(artifact_data);

    var module = try tvm_runtime.RuntimeModule.load_from_file(std.heap.c_allocator, path);
    errdefer module.deinit();

    const main_func = try module.get_function(std.heap.c_allocator, "main", true);
    return .{
        .module = module,
        .main_func = main_func,
    };
}
