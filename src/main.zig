const std = @import("std");
const zg = @import("zigrad");

pub fn main() !void {
    var gpa_state = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa_state.deinit();
    const gpa = gpa_state.allocator();

    var arg_it = std.process.args();
    _ = arg_it.next(); // argv0
    var mode: ?[]const u8 = null;
    var dump_pr_cfg: zg.pipeline.DumpConfig = .{};
    var dump_mlir_cfg: zg.pipeline.DumpConfig = .{};
    var have_dump_pr = false;
    var have_dump_mlir = false;

    while (arg_it.next()) |arg| {
        if (std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "--help")) {
            try print_usage();
            return;
        }
        if (std.mem.eql(u8, arg, "--dump-pr")) {
            dump_pr_cfg = .{ .target = .stdout };
            have_dump_pr = true;
            continue;
        }
        if (std.mem.startsWith(u8, arg, "--dump-pr=")) {
            const path = arg["--dump-pr=".len..];
            if (path.len == 0) {
                try print_usage();
                return error.InvalidArguments;
            }
            dump_pr_cfg = .{ .target = .file, .path = path };
            have_dump_pr = true;
            continue;
        }
        if (std.mem.eql(u8, arg, "--dump-mlir")) {
            dump_mlir_cfg = .{ .target = .stdout };
            have_dump_mlir = true;
            continue;
        }
        if (std.mem.startsWith(u8, arg, "--dump-mlir=")) {
            const path = arg["--dump-mlir=".len..];
            if (path.len == 0) {
                try print_usage();
                return error.InvalidArguments;
            }
            dump_mlir_cfg = .{ .target = .file, .path = path };
            have_dump_mlir = true;
            continue;
        }
        if (std.mem.startsWith(u8, arg, "--")) {
            try print_usage();
            return error.InvalidArguments;
        }
        if (mode == null) {
            mode = arg;
        } else {
            try print_usage();
            return error.InvalidArguments;
        }
    }

    if (mode) |m| {
        if (std.mem.eql(u8, m, "print-pr")) {
            return print_pr(gpa);
        }
    }

    const plugin_path = std.process.getEnvVarOwned(gpa, "PJRT_PLUGIN_PATH") catch |err| {
        std.log.err("PJRT_PLUGIN_PATH not set ({s})", .{@errorName(err)});
        return err;
    };
    defer gpa.free(plugin_path);

    // Initialize unified PJRT backend
    var backend = try zg.backend.PjrtBackend.init(gpa, plugin_path);
    defer backend.deinit();

    const devs = try backend.get_devices(gpa);
    defer gpa.free(devs);
    if (devs.len == 0) return error.NoDevices;
    const device = &devs[0];

    if (mode) |m| {
        if (std.mem.eql(u8, m, "custom-call-neg")) {
            const emit_text = have_dump_mlir;
            return run_custom_call_negative(gpa, &backend, device, emit_text);
        }
        if (std.mem.eql(u8, m, "vjp-demo")) {
            const emit_text = have_dump_mlir;
            return run_vjp_demo(gpa, &backend, device, emit_text);
        }
        if (std.mem.eql(u8, m, "jit-cache-save")) {
            const path = arg_it.next() orelse {
                try print_usage();
                return error.InvalidArguments;
            };

            var program = try zg.frontend.build_demo_program(gpa);
            defer program.deinit();
            // Lower PR -> MLIR
            const mlir_bytes = try zg.lower.lower_program_to_mlir(gpa, &program, "main", .mlir_bytecode);
            defer gpa.free(mlir_bytes);

            // Compile and serialize
            const serialized = try backend.compile_serialized(device, mlir_bytes, .mlir_bytecode, .{});
            defer gpa.free(serialized);

            try write_bytes_to_path(path, serialized);
            std.log.info("wrote PJRT JIT cache artifact: {d} bytes -> {s}", .{ serialized.len, path });
            return;
        }
        if (std.mem.eql(u8, m, "jit-cache-run")) {
            const path = arg_it.next() orelse {
                try print_usage();
                return error.InvalidArguments;
            };

            const serialized = try read_bytes_from_path(gpa, path);
            defer gpa.free(serialized);

            var exe = try backend.load_serialized_executable(serialized, null);
            defer exe.deinit();

            return run_demo_executable(gpa, &backend, device, &exe);
        }
        std.log.err("unknown mode: {s}", .{m});
        try print_usage();
        return error.InvalidArguments;
    }

    var program = try zg.frontend.build_demo_program(gpa);
    defer program.deinit();

    const lower_encoding: zg.pipeline.MlirEncoding = if (have_dump_mlir) .text else .bytecode;
    var exe = try compile_program(&backend, gpa, &program, device, .{
        .encoding = lower_encoding,
        .entry_name = "main",
    }, if (have_dump_pr) &dump_pr_cfg else null, if (have_dump_mlir) &dump_mlir_cfg else null);
    defer exe.deinit();

    return run_demo_executable(gpa, &backend, device, &exe);
}

fn write_bytes_to_path(path: []const u8, bytes: []const u8) !void {
    var file = if (std.fs.path.isAbsolute(path))
        try std.fs.createFileAbsolute(path, .{ .truncate = true })
    else
        try std.fs.cwd().createFile(path, .{ .truncate = true });
    defer file.close();
    try file.writeAll(bytes);
}

fn read_bytes_from_path(allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    var file = if (std.fs.path.isAbsolute(path))
        try std.fs.openFileAbsolute(path, .{})
    else
        try std.fs.cwd().openFile(path, .{});
    defer file.close();
    return file.readToEndAlloc(allocator, std.math.maxInt(usize));
}

fn run_demo_executable(
    allocator: std.mem.Allocator,
    backend: *zg.backend.PjrtBackend,
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

    const shape_a = zg.utils.Shape{ .dims = &.{ 2, 3 } };
    const shape_b = zg.utils.Shape{ .dims = &.{ 3, 2 } };
    const shape_c = zg.utils.Shape{ .dims = &.{ 2, 2 } };

    var host_a = try zg.utils.HostBuffer.from_slice(allocator, &A, shape_a, .f32);
    defer host_a.deinit();
    var host_b = try zg.utils.HostBuffer.from_slice(allocator, &B, shape_b, .f32);
    defer host_b.deinit();
    var host_c = try zg.utils.HostBuffer.from_slice(allocator, &C, shape_c, .f32);
    defer host_c.deinit();

    const dims_a = [_]i64{ 2, 3 };
    const dims_b = [_]i64{ 3, 2 };
    const dims_c = [_]i64{ 2, 2 };

    var dev_a = try backend.buffer_from_host(device, host_a.data, .f32, dims_a[0..]);
    defer dev_a.deinit();
    var dev_b = try backend.buffer_from_host(device, host_b.data, .f32, dims_b[0..]);
    defer dev_b.deinit();
    var dev_c = try backend.buffer_from_host(device, host_c.data, .f32, dims_c[0..]);
    defer dev_c.deinit();

    const outputs = try exe.execute(allocator, &.{ dev_a, dev_b, dev_c });
    defer {
        for (outputs) |*buf| buf.deinit();
        allocator.free(outputs);
    }

    if (outputs.len != 1) return error.UnexpectedOutputs;

    var out_host = try zg.utils.HostBuffer.init(allocator, shape_c, .f32);
    defer out_host.deinit();
    var ev = try outputs[0].to_host(out_host.data);
    defer ev.deinit();
    try ev.await_();

    const out = out_host.as_slice(f32)[0..4];
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

fn run_custom_call_negative(allocator: std.mem.Allocator, backend: *zg.backend.PjrtBackend, device: anytype, emit_text: bool) !void {
    var program = zg.pr.Program.init(allocator);
    defer program.deinit();

    var b = try zg.pr.FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const x = try b.param_tensor(.f32, &.{ 2, 3 });
    const y = try b.custom_call("zigrad.test.missing_handler", &.{x}, x);
    const func = try b.finish(&.{y});
    try program.add_function(func);

    var exe = compile_program(backend, allocator, &program, device, .{
        .encoding = if (emit_text) .text else .bytecode,
        .entry_name = "main",
    }, null, null) catch |err| {
        std.log.info("OK: custom_call compile failed as expected: {s}", .{@errorName(err)});
        return;
    };
    defer exe.deinit();

    std.log.err("unexpected: custom_call compiled without a handler", .{});
    return error.UnexpectedSuccess;
}

fn run_vjp_demo(allocator: std.mem.Allocator, backend: *zg.backend.PjrtBackend, device: anytype, emit_text: bool) !void {
    var program = try zg.frontend.build_demo_program(allocator);
    defer program.deinit();

    const fwd = program.functions[0];
    const vjp = try zg.pr.ad.vjp(allocator, &program, fwd, "main_vjp");
    try program.add_function(vjp);

    var exe = try compile_program(backend, allocator, &program, device, .{
        .encoding = if (emit_text) .text else .bytecode,
        .entry_name = "main_vjp",
    }, null, null);
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

    const shape_a = zg.utils.Shape{ .dims = &.{ 2, 3 } };
    const shape_b = zg.utils.Shape{ .dims = &.{ 3, 2 } };
    const shape_c = zg.utils.Shape{ .dims = &.{ 2, 2 } };

    var host_a = try zg.utils.HostBuffer.from_slice(allocator, &A, shape_a, .f32);
    defer host_a.deinit();
    var host_b = try zg.utils.HostBuffer.from_slice(allocator, &B, shape_b, .f32);
    defer host_b.deinit();
    var host_c = try zg.utils.HostBuffer.from_slice(allocator, &C, shape_c, .f32);
    defer host_c.deinit();
    var host_ct = try zg.utils.HostBuffer.from_slice(allocator, &CtOut, shape_c, .f32);
    defer host_ct.deinit();

    const dims_a = [_]i64{ 2, 3 };
    const dims_b = [_]i64{ 3, 2 };
    const dims_c = [_]i64{ 2, 2 };

    var dev_a = try backend.buffer_from_host(device, host_a.data, .f32, dims_a[0..]);
    defer dev_a.deinit();
    var dev_b = try backend.buffer_from_host(device, host_b.data, .f32, dims_b[0..]);
    defer dev_b.deinit();
    var dev_c = try backend.buffer_from_host(device, host_c.data, .f32, dims_c[0..]);
    defer dev_c.deinit();
    var dev_ct = try backend.buffer_from_host(device, host_ct.data, .f32, dims_c[0..]);
    defer dev_ct.deinit();

    const outputs = try exe.execute(allocator, &.{ dev_a, dev_b, dev_c, dev_ct });
    defer {
        for (outputs) |*buf| buf.deinit();
        allocator.free(outputs);
    }

    if (outputs.len != 3) return error.UnexpectedOutputs;

    var out_a = try zg.utils.HostBuffer.init(allocator, shape_a, .f32);
    defer out_a.deinit();
    var out_b = try zg.utils.HostBuffer.init(allocator, shape_b, .f32);
    defer out_b.deinit();
    var out_c = try zg.utils.HostBuffer.init(allocator, shape_c, .f32);
    defer out_c.deinit();

    var ev_a = try outputs[0].to_host(out_a.data);
    defer ev_a.deinit();
    var ev_b = try outputs[1].to_host(out_b.data);
    defer ev_b.deinit();
    var ev_c = try outputs[2].to_host(out_c.data);
    defer ev_c.deinit();

    try ev_a.await_();
    try ev_b.await_();
    try ev_c.await_();

    const got_a = out_a.as_slice(f32)[0..6];
    const got_b = out_b.as_slice(f32)[0..6];
    const got_c = out_c.as_slice(f32)[0..4];

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

    try expect_all_close("dA", got_a, expected_a[0..], 1e-4);
    try expect_all_close("dB", got_b, expected_b[0..], 1e-4);
    try expect_all_close("dC", got_c, expected_c[0..], 1e-4);

    std.log.info("OK: vjp-demo gradients match expected", .{});
}

fn compile_program(
    backend: *zg.backend.PjrtBackend,
    allocator: std.mem.Allocator,
    program: *zg.pr.Program,
    device: *const zg.backend.pjrt.Device,
    lower_cfg: zg.lower.LowerPassConfig,
    dump_pr: ?*zg.pipeline.DumpConfig,
    dump_mlir: ?*zg.pipeline.DumpConfig,
) !zg.backend.pjrt.LoadedExecutable {
    var lower_cfg_mut = lower_cfg;
    var compile_cfg = zg.backend.pjrt.Backend.CompilePassConfig{ .device = device };

    var passes = std.ArrayList(zg.pipeline.Pass).initCapacity(allocator, 5) catch
        return error.OutOfMemory;
    defer passes.deinit(allocator);

    var dump_pr_local: ?zg.pipeline.DumpConfig = null;
    if (dump_pr) |cfg| {
        dump_pr_local = cfg.*;
        dump_pr_local.?.entry_name = dump_pr_local.?.entry_name orelse lower_cfg.entry_name;
        try passes.append(allocator, zg.pipeline.dump_pr_pass_with_config(&dump_pr_local.?));
    }
    try passes.append(allocator, zg.lower.validate_pass);
    try passes.append(allocator, zg.lower.lower_pass_with_config(&lower_cfg_mut));

    var dump_mlir_local: ?zg.pipeline.DumpConfig = null;
    if (dump_mlir) |cfg| {
        dump_mlir_local = cfg.*;
        dump_mlir_local.?.entry_name = dump_mlir_local.?.entry_name orelse lower_cfg.entry_name;
        try passes.append(allocator, zg.pipeline.dump_mlir_pass_with_config(&dump_mlir_local.?));
    }
    try passes.append(allocator, backend.compile_pass(&compile_cfg));

    const pipeline = zg.pipeline.Pipeline{ .passes = passes.items };

    var ctx = zg.pipeline.PassContext{
        .allocator = allocator,
    };

    var artifact = try pipeline.run(.{ .pr = program }, &ctx);
    errdefer artifact.deinit(allocator);
    return switch (artifact) {
        .ea => |ea| switch (ea) {
            .pjrt => |exe| exe,
        },
        inline else => error.UnexpectedArtifact,
    };
}

fn expect_all_close(label: []const u8, got: []const f32, expected: []const f32, tol: f32) !void {
    if (got.len != expected.len) return error.LengthMismatch;
    for (got, 0..) |v, i| {
        const diff = @abs(v - expected[i]);
        if (diff > tol) {
            std.log.err("{s} mismatch[{d}]: got {d}, expected {d}", .{ label, i, v, expected[i] });
            return error.NumericalMismatch;
        }
    }
}

fn print_pr(allocator: std.mem.Allocator) !void {
    var program = try zg.frontend.build_demo_program(allocator);
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

fn print_usage() !void {
    var buffer: [8192]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&buffer);
    const out = &stdout_writer.interface;

    try out.writeAll(
        \\usage: zigrad [mode] [options]
        \\
        \\modes:
        \\  print-pr
        \\  custom-call-neg
        \\  vjp-demo
        \\  jit-cache-save <path>
        \\  jit-cache-run <path>
        \\
        \\options:
        \\  -h, --help          show this help
        \\  --dump-pr           print PR (zxpr) to stdout
        \\  --dump-pr=PATH      write PR (zxpr) to PATH
        \\  --dump-mlir         print MLIR (text) to stdout
        \\  --dump-mlir=PATH    write MLIR (text) to PATH
        \\
    );

    try out.flush();
}
