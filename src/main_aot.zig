const std = @import("std");
const zg = @import("zigrad");
const demos = @import("demos.zig");

pub fn run(
    allocator: std.mem.Allocator,
    backend: *zg.backend.pjrt.Backend,
    device: *const zg.backend.pjrt.Device,
) !void {
    var program = try demos.build_demo_program(allocator);
    defer program.deinit();

    const mlir_bytes = try zg.lower.lower_program_to_mlir(
        allocator,
        &program,
        "main",
        .mlir_bytecode,
    );
    defer allocator.free(mlir_bytes);

    // Compile, serialize, then reload -- exercises the AOT round-trip.
    const serialized = try backend.compile_serialized(device, mlir_bytes, true, .{});
    defer allocator.free(serialized);

    var loaded = try backend.load_serialized_executable(serialized, null);
    defer backend.deinit_executable(&loaded);

    try run_demo_executable(allocator, backend, device, &loaded);

    std.log.info("OK: aot-demo executed", .{});
}

fn run_demo_executable(
    allocator: std.mem.Allocator,
    backend: *zg.backend.pjrt.Backend,
    device: *const zg.backend.pjrt.Device,
    exe: *zg.backend.pjrt.LoadedExecutable,
) !void {
    // Inputs (A: 2x3, B: 3x2, C: 2x2)
    const A = [_]f32{
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
    };
    const B = [_]f32{
        7.0,  8.0,
        9.0,  10.0,
        11.0, 12.0,
    };
    const C = [_]f32{
        2.0, 2.0,
        2.0, 2.0,
    };

    const shape_a = zg.BoundedShape.from_slice(&.{ 2, 3 });
    const shape_b = zg.BoundedShape.from_slice(&.{ 3, 2 });
    const shape_c = zg.BoundedShape.from_slice(&.{ 2, 2 });

    var host_a = try zg.HostBuffer.from_slice(allocator, &A, shape_a, .f32);
    defer host_a.deinit();
    var host_b = try zg.HostBuffer.from_slice(allocator, &B, shape_b, .f32);
    defer host_b.deinit();
    var host_c = try zg.HostBuffer.from_slice(allocator, &C, shape_c, .f32);
    defer host_c.deinit();

    const dims_a = [_]i64{ 2, 3 };
    const dims_b = [_]i64{ 3, 2 };
    const dims_c = [_]i64{ 2, 2 };

    var dev_a = try backend.buffer_from_host(device, host_a.data(), .f32, dims_a[0..]);
    defer backend.deinit_buffer(&dev_a);
    var dev_b = try backend.buffer_from_host(device, host_b.data(), .f32, dims_b[0..]);
    defer backend.deinit_buffer(&dev_b);
    var dev_c = try backend.buffer_from_host(device, host_c.data(), .f32, dims_c[0..]);
    defer backend.deinit_buffer(&dev_c);

    const result = try backend.execute(exe, allocator, &.{ dev_a, dev_b, dev_c }, .{});
    defer {
        if (result.device_complete_event) |ev| {
            var tmp = ev;
            backend.deinit_event(&tmp);
        }
        for (result.outputs) |*buf| backend.deinit_buffer(buf);
        allocator.free(result.outputs);
    }

    if (result.outputs.len != 1) return error.UnexpectedOutputs;

    var out_host = try zg.HostBuffer.init(allocator, shape_c, .f32);
    defer out_host.deinit();
    var ev = try backend.buffer_to_host(&result.outputs[0], out_host.data_mut());
    defer backend.deinit_event(&ev);
    try backend.await_event(&ev);

    const out = out_host.as_slice(f32);
    std.debug.assert(out.len == 4);
    const expected = [_]f32{
        120.0, 132.0,
        282.0, 312.0,
    };

    for (out, 0..) |v, i| {
        const diff = @abs(v - expected[i]);
        if (diff > 1e-4) {
            std.log.err("mismatch[{d}]: got {d}, expected {d}", .{ i, v, expected[i] });
            return error.NumericalMismatch;
        }
    }

    std.log.info("OK: demo output matches expected", .{});
}
