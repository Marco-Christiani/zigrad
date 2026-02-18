/// TVM matmul adapter for benchmark harness.
///
/// Bridges the benchmark harness to TVM compiled kernels via DLPack tensors
/// and the typed TVM FFI wrappers. CPU path borrows host memory via DLPack;
/// GPU path allocates device tensors and copies data.
const std = @import("std");
const zg = @import("../root.zig");
const tvm_c = @import("../ffi/tvm/c.zig");
const dlpack = zg.tvm_ffi.dlpack;
const tvm_api = zg.tvm_ffi.tvm_api;
const tvm_types = zg.tvm_ffi.tvm_types;
const tvm_runtime = zg.tvm_runtime;
const build_options = @import("build_options");

/// Execute TVM CPU matmul using pre-loaded module (for cached execution).
pub fn execute_with_module(
    allocator: std.mem.Allocator,
    tuned: *tvm_runtime.TunedModule,
    m: usize,
    n: usize,
    k: usize,
    a: []const f32,
    b: []const f32,
    c: []f32,
) !void {
    if (!build_options.enable_tvm) return error.TvmDisabled;

    // Create DLPack tensors borrowing existing host buffers
    var shape_a = [_]i64{ @intCast(m), @intCast(k) };
    var shape_b = [_]i64{ @intCast(k), @intCast(n) };
    var shape_c = [_]i64{ @intCast(m), @intCast(n) };

    var dl_a = dlpack.ManagedTensor.borrowing(
        dlpack.Tensor.init_contiguous(f32, @constCast(a), &shape_a),
    );
    var dl_b = dlpack.ManagedTensor.borrowing(
        dlpack.Tensor.init_contiguous(f32, @constCast(b), &shape_b),
    );
    var dl_c = dlpack.ManagedTensor.borrowing(
        dlpack.Tensor.init_contiguous(f32, c, &shape_c),
    );

    // Convert to TVM tensors
    var t_a = try tvm_types.Tensor.from_dlpack(&dl_a);
    defer t_a.deinit();
    var t_b = try tvm_types.Tensor.from_dlpack(&dl_b);
    defer t_b.deinit();
    var t_c = try tvm_types.Tensor.from_dlpack(&dl_c);
    defer t_c.deinit();

    // Execute
    var call_args = [_]tvm_c.TVMFFIAny{
        t_a.as_value().raw,
        t_b.as_value().raw,
        t_c.as_value().raw,
    };
    var call_res = tvm_api.Value.none().raw;

    try tvm_api.call(allocator, tuned.main_func, &call_args, &call_res);
}

/// Execute TVM GPU matmul using pre-loaded module (for cached execution).
pub fn execute_gpu_with_module(
    allocator: std.mem.Allocator,
    tuned: *tvm_runtime.TunedModule,
    m: usize,
    n: usize,
    k: usize,
    a: []const f32,
    b: []const f32,
    c: []f32,
) !void {
    if (!build_options.enable_tvm) return error.TvmDisabled;

    // Allocate GPU tensors (includes host->device copy)
    var shape_a = [_]i64{ @intCast(m), @intCast(k) };
    var shape_b = [_]i64{ @intCast(k), @intCast(n) };
    var shape_c = [_]i64{ @intCast(m), @intCast(n) };

    var t_a = try tvm_types.Tensor.allocate(allocator, @constCast(a), &shape_a, .cuda);
    defer t_a.deinit();
    var t_b = try tvm_types.Tensor.allocate(allocator, @constCast(b), &shape_b, .cuda);
    defer t_b.deinit();

    // Allocate output tensor on GPU (zero-initialized)
    const c_init = try allocator.alloc(f32, m * n);
    defer allocator.free(c_init);
    @memset(c_init, 0);
    var t_c = try tvm_types.Tensor.allocate(allocator, c_init, &shape_c, .cuda);
    defer t_c.deinit();

    // Execute kernel on GPU
    var call_args = [_]tvm_c.TVMFFIAny{
        t_a.as_value().raw,
        t_b.as_value().raw,
        t_c.as_value().raw,
    };
    var call_res = tvm_api.Value.none().raw;

    try tvm_api.call(allocator, tuned.main_func, &call_args, &call_res);

    // Copy result back to host
    try t_c.copy_to_host(allocator, c);
}
