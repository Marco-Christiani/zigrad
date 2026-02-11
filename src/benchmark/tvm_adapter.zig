/// TVM CPU matmul adapter for benchmark harness.
const std = @import("std");
const tvm = @import("../tvm_runtime.zig");
const build_options = @import("build_options");

/// Execute TVM CPU matmul using pre-loaded module (for cached execution).
pub fn execute_with_module(
    allocator: std.mem.Allocator,
    tuned: *tvm.TunedModule,
    m: usize,
    n: usize,
    k: usize,
    a: []const f32,
    b: []const f32,
    c: []f32,
) !void {
    if (!build_options.enable_tvm) return error.TvmDisabled;

    // Create DLPack tensors from existing buffers
    var shape_a = [_]i64{ @intCast(m), @intCast(k) };
    var shape_b = [_]i64{ @intCast(k), @intCast(n) };
    var shape_c = [_]i64{ @intCast(m), @intCast(n) };

    var dl_a = tvm.c.DLManagedTensor{
        .dl_tensor = tvm.make_dl_tensor_f32(@constCast(a), &shape_a),
        .manager_ctx = null,
        .deleter = tvm.dlpack_noop_deleter,
    };
    var dl_b = tvm.c.DLManagedTensor{
        .dl_tensor = tvm.make_dl_tensor_f32(@constCast(b), &shape_b),
        .manager_ctx = null,
        .deleter = tvm.dlpack_noop_deleter,
    };
    var dl_c = tvm.c.DLManagedTensor{
        .dl_tensor = tvm.make_dl_tensor_f32(c, &shape_c),
        .manager_ctx = null,
        .deleter = tvm.dlpack_noop_deleter,
    };

    // Convert to TVM tensors
    const t_a = try tvm.tensor_from_dlpack(allocator, &dl_a);
    defer _ = tvm.c.TVMFFIObjectDecRef(t_a);
    const t_b = try tvm.tensor_from_dlpack(allocator, &dl_b);
    defer _ = tvm.c.TVMFFIObjectDecRef(t_b);
    const t_c = try tvm.tensor_from_dlpack(allocator, &dl_c);
    defer _ = tvm.c.TVMFFIObjectDecRef(t_c);

    // Execute
    var call_args = [_]tvm.c.TVMFFIAny{
        tvm.any_obj(t_a, tvm.c.kTVMFFITensor),
        tvm.any_obj(t_b, tvm.c.kTVMFFITensor),
        tvm.any_obj(t_c, tvm.c.kTVMFFITensor),
    };
    var call_res: tvm.c.TVMFFIAny = std.mem.zeroes(tvm.c.TVMFFIAny);

    try tvm.ffi_call(allocator, tuned.main_func, &call_args, &call_res);
}

/// Execute TVM GPU matmul using pre-loaded module (for cached execution).
pub fn execute_gpu_with_module(
    allocator: std.mem.Allocator,
    tuned: *tvm.TunedModule,
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

    const t_a = try tvm.allocate_tensor(allocator, @constCast(a), &shape_a, tvm.c.kDLCUDA);
    defer _ = tvm.c.TVMFFIObjectDecRef(t_a);
    const t_b = try tvm.allocate_tensor(allocator, @constCast(b), &shape_b, tvm.c.kDLCUDA);
    defer _ = tvm.c.TVMFFIObjectDecRef(t_b);

    // Allocate output tensor on GPU (zero-initialized)
    const c_init = try allocator.alloc(f32, m * n);
    defer allocator.free(c_init);
    @memset(c_init, 0);
    const t_c = try tvm.allocate_tensor(allocator, c_init, &shape_c, tvm.c.kDLCUDA);
    defer _ = tvm.c.TVMFFIObjectDecRef(t_c);

    // Execute kernel on GPU
    var call_args = [_]tvm.c.TVMFFIAny{
        tvm.any_obj(t_a, tvm.c.kTVMFFITensor),
        tvm.any_obj(t_b, tvm.c.kTVMFFITensor),
        tvm.any_obj(t_c, tvm.c.kTVMFFITensor),
    };
    var call_res: tvm.c.TVMFFIAny = std.mem.zeroes(tvm.c.TVMFFIAny);

    try tvm.ffi_call(allocator, tuned.main_func, &call_args, &call_res);

    // Copy result back to host
    try tvm.copy_tensor_to_host(allocator, t_c, c);
}
