/// Minimal IREE VMFB runner.
///
/// Loads a pre-compiled VMFB and executes `module.main` with hardcoded
/// demo inputs (A=2x3, B=3x2, C=2x2 f32 matmul+add+mul).
///
/// This binary does NOT depend on zigrad, MLIR, PJRT, or TVM.
/// It links only the IREE runtime static archives + libc.
///
/// Usage: iree-runner <vmfb-path> [hal-driver]
///   hal-driver defaults to "local-sync"
const std = @import("std");
const rt = @import("c/iree/runtime.zig");

const log = std.log.scoped(.@"zg/iree_runner");

pub fn main() !void {
    const gpa = std.heap.smp_allocator;

    var args = try std.process.argsWithAllocator(gpa);
    defer args.deinit();

    // Skip argv[0].
    _ = args.next();

    const vmfb_path = args.next() orelse {
        std.debug.print("usage: iree-runner <vmfb-path> [hal-driver]\n", .{});
        return error.MissingArgument;
    };
    const driver = args.next() orelse "local-sync";

    // Read VMFB from file.
    const vmfb = blk: {
        var file = if (std.fs.path.isAbsolute(vmfb_path))
            try std.fs.openFileAbsolute(vmfb_path, .{})
        else
            try std.fs.cwd().openFile(vmfb_path, .{});
        defer file.close();
        break :blk try file.readToEndAlloc(gpa, 256 * 1024 * 1024);
    };
    defer gpa.free(vmfb);

    log.info("loaded {d} bytes from {s}", .{ vmfb.len, vmfb_path });

    // Create IREE runtime instance + device.
    const instance = try rt.instance_create();
    defer rt.instance_release(instance);

    const device = try rt.create_default_device(instance, driver);
    defer rt.device_release(device);

    // Create session and load VMFB.
    const session = try rt.session_create(instance, device);
    defer rt.session_release(session);
    try rt.session_append_module(session, vmfb);

    // Look up entry function.
    const function = try rt.session_lookup_function(session, "module.main");

    // Build demo inputs: A(2x3) x B(3x2) + C(2x2), then * C.
    const A = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    const B = [_]f32{ 7.0, 8.0, 9.0, 10.0, 11.0, 12.0 };
    const C = [_]f32{ 2.0, 2.0, 2.0, 2.0 };

    const f32_type = rt.c.IREE_HAL_ELEMENT_TYPE_FLOAT_32;

    const buf_a = try rt.buffer_view_create_from_host(
        device,
        std.mem.asBytes(&A),
        f32_type,
        &[_]rt.HalDim{ 2, 3 },
    );
    defer rt.buffer_view_release(buf_a);

    const buf_b = try rt.buffer_view_create_from_host(
        device,
        std.mem.asBytes(&B),
        f32_type,
        &[_]rt.HalDim{ 3, 2 },
    );
    defer rt.buffer_view_release(buf_b);

    const buf_c = try rt.buffer_view_create_from_host(
        device,
        std.mem.asBytes(&C),
        f32_type,
        &[_]rt.HalDim{ 2, 2 },
    );
    defer rt.buffer_view_release(buf_c);

    // Execute.
    var call = try rt.call_init(session, function);
    defer rt.call_deinit(&call);

    try rt.list_push_buffer_view(rt.call_inputs(&call), buf_a);
    try rt.list_push_buffer_view(rt.call_inputs(&call), buf_b);
    try rt.list_push_buffer_view(rt.call_inputs(&call), buf_c);

    try rt.call_invoke(&call);

    // Read output.
    const out_list = rt.call_outputs(&call);
    const n_out = rt.list_size(out_list);
    if (n_out == 0) {
        log.err("no outputs", .{});
        return error.NoOutputs;
    }

    const out_view = try rt.list_get_buffer_view(out_list, 0);
    defer rt.buffer_view_release(out_view);

    var out: [4]f32 = undefined;
    try rt.buffer_view_to_host(out_view, std.mem.asBytes(&out));

    log.info("result: [{d}, {d}, {d}, {d}]", .{ out[0], out[1], out[2], out[3] });

    // Verify: (dot(A, B) + C) * C = [120, 132, 282, 312].
    const expected = [_]f32{ 120.0, 132.0, 282.0, 312.0 };
    for (out, expected) |got, exp| {
        if (@abs(got - exp) > 1e-3) {
            log.err("mismatch got={d} expected={d}", .{ got, exp });
            return error.NumericalMismatch;
        }
    }
    log.info("OK", .{});
}
