//! TVM matmul adapter for benchmark harness.
//!
//! Bridges the benchmark harness to TVM compiled kernels via DLPack tensors
//!  and the typed TVM FFI wrappers. CPU path borrows host memory via DLPack,
//!  GPU path allocates device tensors and copies data.
const std = @import("std");
const zg = @import("zigrad");
const dlpack = zg.tvm.dlpack;
const tvm_runtime = zg.tvm.runtime;
const tvm_module = zg.tvm.module;

const DeviceKind = @import("harness.zig").DeviceKind;

/// Execute TVM matmul using a pre-loaded tuned module.
pub fn execute_with_module(
    comptime T: type,
    allocator: std.mem.Allocator,
    tuned: *tvm_module.TunedModule,
    m: i64,
    n: i64,
    k: i64,
    a: []const T,
    b: []const T,
    c: []T,
    device: DeviceKind,
) !void {
    try zg.tvm.ffi.ensure_loaded(allocator, .{});

    var shape_a = [_]i64{ m, k };
    var shape_b = [_]i64{ k, n };
    var shape_c = [_]i64{ m, n };

    switch (device) {
        .cpu => {
            var dl_a = dlpack.ManagedTensor.borrowing(
                dlpack.Tensor.init_contiguous(T, @constCast(a), &shape_a),
            );
            var dl_b = dlpack.ManagedTensor.borrowing(
                dlpack.Tensor.init_contiguous(T, @constCast(b), &shape_b),
            );
            var dl_c = dlpack.ManagedTensor.borrowing(
                dlpack.Tensor.init_contiguous(T, c, &shape_c),
            );

            var t_a = try tvm_runtime.Tensor.from_dlpack(&dl_a);
            defer t_a.deinit();
            var t_b = try tvm_runtime.Tensor.from_dlpack(&dl_b);
            defer t_b.deinit();
            var t_c = try tvm_runtime.Tensor.from_dlpack(&dl_c);
            defer t_c.deinit();

            try tuned.invoke(allocator, &.{
                t_a.as_value(), t_b.as_value(), t_c.as_value(),
            });
        },
        .gpu => {
            // tvm_runtime.Tensor.allocate is currently f32-only in zigrad's API.
            if (T != f32) return error.UnsupportedDtype;
            const a_f32: []f32 = @constCast(@ptrCast(a));
            const b_f32: []f32 = @constCast(@ptrCast(b));

            var t_a = try tvm_runtime.Tensor.allocate(allocator, a_f32, &shape_a, .cuda);
            defer t_a.deinit();
            var t_b = try tvm_runtime.Tensor.allocate(allocator, b_f32, &shape_b, .cuda);
            defer t_b.deinit();

            const c_init = try allocator.alloc(f32, @intCast(m * n));
            defer allocator.free(c_init);
            @memset(c_init, @as(f32, 0));
            var t_c = try tvm_runtime.Tensor.allocate(allocator, c_init, &shape_c, .cuda);
            defer t_c.deinit();

            try tuned.invoke(allocator, &.{
                t_a.as_value(), t_b.as_value(), t_c.as_value(),
            });

            const c_f32: []f32 = @ptrCast(c);
            try t_c.copy_to_host(allocator, c_f32);
        },
    }
}
