const std = @import("std");
const zg = @import("zigrad");

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

    var rt = try zg.runtime.pjrt.Runtime.init(gpa, plugin_path);
    defer rt.deinit();

    const devs = try rt.devices(gpa);
    defer gpa.free(devs);
    if (devs.len == 0) return error.NoDevices;
    const device = &devs[0];

    if (mode) |m| {
        if (std.mem.eql(u8, m, "print-pr")) {
            return printPr(gpa);
        }
        if (std.mem.eql(u8, m, "custom-call-neg")) {
            return runCustomCallNegative(gpa, &rt, device);
        }
        if (std.mem.eql(u8, m, "vjp-demo")) {
            return runVjpDemo(gpa, &rt, device);
        }
        if (std.mem.eql(u8, m, "jit-cache-save") or std.mem.eql(u8, m, "aot-save")) {
            if (std.mem.eql(u8, m, "aot-save")) {
                std.log.warn("mode aot-save is deprecated; use jit-cache-save", .{});
            }
            const path = arg_it.next() orelse {
                std.log.err("usage: zigrad jit-cache-save <path>", .{});
                return error.InvalidArguments;
            };

            var program = try zg.frontend.buildDemoProgram(gpa);
            defer program.deinit();
            const func = program.functions[0];

            // PR -> IM realization
            var im = try zg.im.stablehlo.realize(gpa, func, .{});
            defer im.deinit();

            const serialized = try zg.toolchain.xla.compileSerialized(gpa, rt.getClient(), device, im, .{});
            defer gpa.free(serialized);

            try writeBytesToPath(path, serialized);
            std.log.info("wrote PJRT serialized executable: {d} bytes -> {s}", .{ serialized.len, path });
            return;
        }
        if (std.mem.eql(u8, m, "jit-cache-run") or std.mem.eql(u8, m, "aot-run")) {
            if (std.mem.eql(u8, m, "aot-run")) {
                std.log.warn("mode aot-run is deprecated; use jit-cache-run", .{});
            }
            const path = arg_it.next() orelse {
                std.log.err("usage: zigrad jit-cache-run <path>", .{});
                return error.InvalidArguments;
            };

            const serialized = try readBytesFromPath(gpa, path);
            defer gpa.free(serialized);

            var exe = try rt.loadSerializedExecutable(serialized, null);
            defer exe.deinit();

            return runDemoExecutable(gpa, &rt, device, &exe);
        }
        std.log.err("unknown mode: {s}", .{m});
        return error.InvalidArguments;
    }

    var program = try zg.frontend.buildDemoProgram(gpa);
    defer program.deinit();

    const func = program.functions[0];

    // PR -> IM realization
    var im = try zg.im.stablehlo.realize(gpa, func, .{});
    defer im.deinit();

    // IM -> EA compilation
    var exe = try zg.toolchain.xla.compile(gpa, rt.getClient(), device, im, .{});
    defer exe.deinit();

    return runDemoExecutable(gpa, &rt, device, &exe);
}

fn writeBytesToPath(path: []const u8, bytes: []const u8) !void {
    var file = if (std.fs.path.isAbsolute(path))
        try std.fs.createFileAbsolute(path, .{ .truncate = true })
    else
        try std.fs.cwd().createFile(path, .{ .truncate = true });
    defer file.close();
    try file.writeAll(bytes);
}

fn readBytesFromPath(allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    var file = if (std.fs.path.isAbsolute(path))
        try std.fs.openFileAbsolute(path, .{})
    else
        try std.fs.cwd().openFile(path, .{});
    defer file.close();
    return file.readToEndAlloc(allocator, std.math.maxInt(usize));
}

fn runDemoExecutable(
    allocator: std.mem.Allocator,
    rt: *zg.runtime.pjrt.Runtime,
    device: *const zg.runtime.pjrt.Device,
    exe: *zg.runtime.pjrt.LoadedExecutable,
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

    const shape_a = zg.runtime.Shape{ .dims = &.{ 2, 3 } };
    const shape_b = zg.runtime.Shape{ .dims = &.{ 3, 2 } };
    const shape_c = zg.runtime.Shape{ .dims = &.{ 2, 2 } };

    var host_a = try zg.runtime.HostBuffer.fromSlice(allocator, &A, shape_a, .f32);
    defer host_a.deinit();
    var host_b = try zg.runtime.HostBuffer.fromSlice(allocator, &B, shape_b, .f32);
    defer host_b.deinit();
    var host_c = try zg.runtime.HostBuffer.fromSlice(allocator, &C, shape_c, .f32);
    defer host_c.deinit();

    const dims_a = [_]i64{ 2, 3 };
    const dims_b = [_]i64{ 3, 2 };
    const dims_c = [_]i64{ 2, 2 };

    var dev_a = try rt.bufferFromHost(device, host_a.data, .f32, dims_a[0..]);
    defer dev_a.deinit();
    var dev_b = try rt.bufferFromHost(device, host_b.data, .f32, dims_b[0..]);
    defer dev_b.deinit();
    var dev_c = try rt.bufferFromHost(device, host_c.data, .f32, dims_c[0..]);
    defer dev_c.deinit();

    const outputs = try exe.execute(allocator, &.{ dev_a, dev_b, dev_c });
    defer {
        for (outputs) |*buf| buf.deinit();
        allocator.free(outputs);
    }

    if (outputs.len != 1) return error.UnexpectedOutputs;

    var out_host = try zg.runtime.HostBuffer.init(allocator, shape_c, .f32);
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

fn runCustomCallNegative(allocator: std.mem.Allocator, rt: *zg.runtime.pjrt.Runtime, device: anytype) !void {
    var program = zg.pr.Program.init(allocator);
    defer program.deinit();

    var b = try zg.pr.FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const x = try b.paramTensor(.f32, &.{ 2, 3 });
    const y = try b.customCall("zigrad.test.missing_handler", &.{x}, x);
    const func = try b.finish(&.{y});

    // PR -> IM realization
    var im = try zg.im.stablehlo.realize(allocator, func, .{});
    defer im.deinit();

    // IM -> EA compilation (expected to fail)
    var exe = zg.toolchain.xla.compile(allocator, rt.getClient(), device, im, .{}) catch |err| {
        std.log.info("OK: custom_call compile failed as expected: {s}", .{@errorName(err)});
        return;
    };
    defer exe.deinit();

    std.log.err("unexpected: custom_call compiled without a handler", .{});
    return error.UnexpectedSuccess;
}

fn runVjpDemo(allocator: std.mem.Allocator, rt: *zg.runtime.pjrt.Runtime, device: anytype) !void {
    var program = try zg.frontend.buildDemoProgram(allocator);
    defer program.deinit();

    const fwd = program.functions[0];
    const vjp = try zg.pr.ad.vjp(allocator, &program, fwd, "main_vjp");

    // PR -> IM realization
    var im = try zg.im.stablehlo.realize(allocator, vjp, .{});
    defer im.deinit();

    // IM -> EA compilation
    var exe = try zg.toolchain.xla.compile(allocator, rt.getClient(), device, im, .{});
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

    const shape_a = zg.runtime.Shape{ .dims = &.{ 2, 3 } };
    const shape_b = zg.runtime.Shape{ .dims = &.{ 3, 2 } };
    const shape_c = zg.runtime.Shape{ .dims = &.{ 2, 2 } };

    var host_a = try zg.runtime.HostBuffer.fromSlice(allocator, &A, shape_a, .f32);
    defer host_a.deinit();
    var host_b = try zg.runtime.HostBuffer.fromSlice(allocator, &B, shape_b, .f32);
    defer host_b.deinit();
    var host_c = try zg.runtime.HostBuffer.fromSlice(allocator, &C, shape_c, .f32);
    defer host_c.deinit();
    var host_ct = try zg.runtime.HostBuffer.fromSlice(allocator, &CtOut, shape_c, .f32);
    defer host_ct.deinit();

    const dims_a = [_]i64{ 2, 3 };
    const dims_b = [_]i64{ 3, 2 };
    const dims_c = [_]i64{ 2, 2 };

    var dev_a = try rt.bufferFromHost(device, host_a.data, .f32, dims_a[0..]);
    defer dev_a.deinit();
    var dev_b = try rt.bufferFromHost(device, host_b.data, .f32, dims_b[0..]);
    defer dev_b.deinit();
    var dev_c = try rt.bufferFromHost(device, host_c.data, .f32, dims_c[0..]);
    defer dev_c.deinit();
    var dev_ct = try rt.bufferFromHost(device, host_ct.data, .f32, dims_c[0..]);
    defer dev_ct.deinit();

    const outputs = try exe.execute(allocator, &.{ dev_a, dev_b, dev_c, dev_ct });
    defer {
        for (outputs) |*buf| buf.deinit();
        allocator.free(outputs);
    }

    if (outputs.len != 3) return error.UnexpectedOutputs;

    var out_a = try zg.runtime.HostBuffer.init(allocator, shape_a, .f32);
    defer out_a.deinit();
    var out_b = try zg.runtime.HostBuffer.init(allocator, shape_b, .f32);
    defer out_b.deinit();
    var out_c = try zg.runtime.HostBuffer.init(allocator, shape_c, .f32);
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
        62.0,  68.0,
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

fn printPr(allocator: std.mem.Allocator) !void {
    var program = try zg.frontend.buildDemoProgram(allocator);
    defer program.deinit();

    const fwd = program.functions[0];
    const vjp_func = try zg.pr.ad.vjp(allocator, &program, fwd, "main_vjp");

    var stdout_buffer: [8192]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    const stdout = &stdout_writer.interface;
    defer stdout.flush() catch {};

    try stdout.writeAll("=== Forward ===\n");
    try zg.pr.zxpr.emit(fwd, stdout);
    try stdout.writeAll("\n=== VJP ===\n");
    try zg.pr.zxpr.emit(vjp_func, stdout);
}
