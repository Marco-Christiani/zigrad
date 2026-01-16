const std = @import("std");
const zigrad = @import("zigrad");

pub fn main() !void {
    var gpa_state = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa_state.deinit();
    const gpa = gpa_state.allocator();

    var arg_it = std.process.args();
    _ = arg_it.next(); // argv0
    const mode = arg_it.next();

    const plugin_path = std.process.getEnvVarOwned(gpa, "PJRT_PLUGIN_PATH") catch |err| {
        std.log.err("PJRT_PLUGIN_PATH not set ({s})", .{@errorName(err)});
        return err;
    };
    defer gpa.free(plugin_path);

    var rt = try zigrad.runtime.pjrt.Runtime.init(gpa, plugin_path);
    defer rt.deinit();

    const devs = try rt.devices(gpa);
    defer gpa.free(devs);
    if (devs.len == 0) return error.NoDevices;
    const device = &devs[0];

    if (mode) |m| {
        if (std.mem.eql(u8, m, "custom-call-neg")) {
            return runCustomCallNegative(gpa, &rt, device);
        }
        if (std.mem.eql(u8, m, "vjp-demo")) {
            return runVjpDemo(gpa, &rt, device);
        }
        std.log.err("unknown mode: {s}", .{m});
        return error.InvalidArguments;
    }

    var program = try zigrad.frontend.buildDemoProgram(gpa);
    defer program.deinit();

    const func = program.functions[0];
    var exe = try zigrad.toolchain.xla.compileJit(gpa, rt.getClient(), device, func);
    defer exe.deinit();

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

    const shape_a = zigrad.runtime.Shape{ .dims = &.{ 2, 3 } };
    const shape_b = zigrad.runtime.Shape{ .dims = &.{ 3, 2 } };
    const shape_c = zigrad.runtime.Shape{ .dims = &.{ 2, 2 } };

    var host_a = try zigrad.runtime.HostBuffer.fromSlice(gpa, &A, shape_a, .f32);
    defer host_a.deinit();
    var host_b = try zigrad.runtime.HostBuffer.fromSlice(gpa, &B, shape_b, .f32);
    defer host_b.deinit();
    var host_c = try zigrad.runtime.HostBuffer.fromSlice(gpa, &C, shape_c, .f32);
    defer host_c.deinit();

    const dims_a = [_]i64{ 2, 3 };
    const dims_b = [_]i64{ 3, 2 };
    const dims_c = [_]i64{ 2, 2 };

    var dev_a = try rt.client.bufferFromHost(device, host_a.data, .f32, dims_a[0..]);
    defer dev_a.deinit();
    var dev_b = try rt.client.bufferFromHost(device, host_b.data, .f32, dims_b[0..]);
    defer dev_b.deinit();
    var dev_c = try rt.client.bufferFromHost(device, host_c.data, .f32, dims_c[0..]);
    defer dev_c.deinit();

    const outputs = try exe.execute(gpa, &.{ dev_a, dev_b, dev_c });
    defer {
        for (outputs) |*buf| buf.deinit();
        gpa.free(outputs);
    }

    if (outputs.len != 1) return error.UnexpectedOutputs;

    var out_host = try zigrad.runtime.HostBuffer.init(gpa, shape_c, .f32);
    defer out_host.deinit();
    var ev = try outputs[0].toHost(out_host.data);
    defer ev.deinit();
    try ev.await_();

    const out = out_host.asSlice(f32)[0..4];
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

fn runCustomCallNegative(allocator: std.mem.Allocator, rt: *zigrad.runtime.pjrt.Runtime, device: anytype) !void {
    var program = zigrad.pr.Program.init(allocator);
    defer program.deinit();

    var b = try zigrad.pr.FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const x = try b.paramTensor(.f32, &.{ 2, 3 });
    const y = try b.customCall("zigrad.test.missing_handler", &.{x}, x);
    const func = try b.finish(&.{y});

    var exe = zigrad.toolchain.xla.compileJit(allocator, rt.getClient(), device, func) catch |err| {
        std.log.info("OK: custom_call compile failed as expected: {s}", .{@errorName(err)});
        return;
    };
    defer exe.deinit();

    std.log.err("unexpected: custom_call compiled without a handler", .{});
    return error.UnexpectedSuccess;
}

fn runVjpDemo(allocator: std.mem.Allocator, rt: *zigrad.runtime.pjrt.Runtime, device: anytype) !void {
    var program = try zigrad.frontend.buildDemoProgram(allocator);
    defer program.deinit();

    const fwd = program.functions[0];
    const vjp = try zigrad.pr.ad.vjp(allocator, &program, fwd, "main_vjp");

    var exe = try zigrad.toolchain.xla.compileJit(allocator, rt.getClient(), device, vjp);
    defer exe.deinit();

    // Inputs (A: 2x3, B: 3x2, C: 2x2, cotangent(out): 2x2)
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
    const CtOut = [_]f32{
        1.0, 1.0,
        1.0, 1.0,
    };

    const shape_a = zigrad.runtime.Shape{ .dims = &.{ 2, 3 } };
    const shape_b = zigrad.runtime.Shape{ .dims = &.{ 3, 2 } };
    const shape_c = zigrad.runtime.Shape{ .dims = &.{ 2, 2 } };

    var host_a = try zigrad.runtime.HostBuffer.fromSlice(allocator, &A, shape_a, .f32);
    defer host_a.deinit();
    var host_b = try zigrad.runtime.HostBuffer.fromSlice(allocator, &B, shape_b, .f32);
    defer host_b.deinit();
    var host_c = try zigrad.runtime.HostBuffer.fromSlice(allocator, &C, shape_c, .f32);
    defer host_c.deinit();
    var host_ct = try zigrad.runtime.HostBuffer.fromSlice(allocator, &CtOut, shape_c, .f32);
    defer host_ct.deinit();

    const dims_a = [_]i64{ 2, 3 };
    const dims_b = [_]i64{ 3, 2 };
    const dims_c = [_]i64{ 2, 2 };

    var dev_a = try rt.client.bufferFromHost(device, host_a.data, .f32, dims_a[0..]);
    defer dev_a.deinit();
    var dev_b = try rt.client.bufferFromHost(device, host_b.data, .f32, dims_b[0..]);
    defer dev_b.deinit();
    var dev_c = try rt.client.bufferFromHost(device, host_c.data, .f32, dims_c[0..]);
    defer dev_c.deinit();
    var dev_ct = try rt.client.bufferFromHost(device, host_ct.data, .f32, dims_c[0..]);
    defer dev_ct.deinit();

    const outputs = try exe.execute(allocator, &.{ dev_a, dev_b, dev_c, dev_ct });
    defer {
        for (outputs) |*buf| buf.deinit();
        allocator.free(outputs);
    }

    if (outputs.len != 3) return error.UnexpectedOutputs;

    var out_a = try zigrad.runtime.HostBuffer.init(allocator, shape_a, .f32);
    defer out_a.deinit();
    var out_b = try zigrad.runtime.HostBuffer.init(allocator, shape_b, .f32);
    defer out_b.deinit();
    var out_c = try zigrad.runtime.HostBuffer.init(allocator, shape_c, .f32);
    defer out_c.deinit();

    var ev_a = try outputs[0].toHost(out_a.data);
    defer ev_a.deinit();
    var ev_b = try outputs[1].toHost(out_b.data);
    defer ev_b.deinit();
    var ev_c = try outputs[2].toHost(out_c.data);
    defer ev_c.deinit();

    try ev_a.await_();
    try ev_b.await_();
    try ev_c.await_();

    const got_a = out_a.asSlice(f32)[0..6];
    const got_b = out_b.asSlice(f32)[0..6];
    const got_c = out_c.asSlice(f32)[0..4];

    const expected_a = [_]f32{
        30.0, 38.0, 46.0,
        30.0, 38.0, 46.0,
    };
    const expected_b = [_]f32{
        10.0, 10.0,
        14.0, 14.0,
        18.0, 18.0,
    };
    const expected_c = [_]f32{
        62.0, 68.0,
        143.0, 158.0,
    };

    try expectAllClose("dA", got_a, expected_a[0..], 1e-4);
    try expectAllClose("dB", got_b, expected_b[0..], 1e-4);
    try expectAllClose("dC", got_c, expected_c[0..], 1e-4);

    std.log.info("OK: vjp-demo gradients match expected", .{});
}

fn expectAllClose(label: []const u8, got: []const f32, expected: []const f32, tol: f32) !void {
    if (got.len != expected.len) return error.LengthMismatch;
    for (got, 0..) |v, i| {
        const diff = @abs(v - expected[i]);
        if (diff > tol) {
            std.log.err("{s} mismatch[{d}]: got {d}, expected {d}", .{ label, i, v, expected[i] });
            return error.NumericalMismatch;
        }
    }
}
