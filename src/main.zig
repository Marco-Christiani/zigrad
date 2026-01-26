const std = @import("std");
const zg = @import("zigrad");

pub fn main() !void {
    var gpa_state = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa_state.deinit();
    const gpa = gpa_state.allocator();

    var arg_it = std.process.args();
    _ = arg_it.next(); // argv0
    var mode: ?[]const u8 = null;
    var mode_args = std.ArrayList([]const u8).empty;
    defer mode_args.deinit(gpa);
    var dump_pr_cfg: zg.pipeline.DumpConfig = .{};
    var dump_mlir_cfg: zg.pipeline.DumpConfig = .{};
    var have_dump_pr = false;
    var have_dump_mlir = false;

    while (arg_it.next()) |arg| {
        if (std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "--help")) {
            try print_usage();
            return;
        }
        if (mode != null) {
            if (std.mem.startsWith(u8, arg, "-")) {
                try print_usage();
                return error.InvalidArguments;
            }
            try mode_args.append(gpa, arg);
            continue;
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
        mode = arg;
    }

    if (mode) |m| {
        if (std.mem.eql(u8, m, "print-pr")) {
            if (mode_args.items.len != 0 or have_dump_pr or have_dump_mlir) {
                try print_usage();
                return error.InvalidArguments;
            }
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
            if (mode_args.items.len != 0) {
                try print_usage();
                return error.InvalidArguments;
            }
            return run_custom_call_negative(gpa, &backend, device, if (have_dump_pr) &dump_pr_cfg else null, if (have_dump_mlir) &dump_mlir_cfg else null);
        }
        if (std.mem.eql(u8, m, "vjp-demo")) {
            if (mode_args.items.len != 0) {
                try print_usage();
                return error.InvalidArguments;
            }
            return run_vjp_demo(gpa, &backend, device, if (have_dump_pr) &dump_pr_cfg else null, if (have_dump_mlir) &dump_mlir_cfg else null);
        }
        if (std.mem.eql(u8, m, "train-demo")) {
            if (mode_args.items.len > 1) {
                try print_usage();
                return error.InvalidArguments;
            }
            const warmup_steps: usize = if (mode_args.items.len == 1)
                std.fmt.parseInt(usize, mode_args.items[0], 10) catch {
                    try print_usage();
                    return error.InvalidArguments;
                }
            else
                0;
            return run_train_demo(
                gpa,
                plugin_path,
                if (have_dump_pr) &dump_pr_cfg else null,
                if (have_dump_mlir) &dump_mlir_cfg else null,
                warmup_steps,
            );
        }
        if (std.mem.eql(u8, m, "jit-cache-save")) {
            if (have_dump_pr or have_dump_mlir) {
                try print_usage();
                return error.InvalidArguments;
            }
            const path: ?[]const u8 = if (mode_args.items.len == 1) mode_args.items[0] else null;
            if (path == null) {
                try print_usage();
                return error.InvalidArguments;
            }

            var program = try zg.frontend.build_demo_program(gpa);
            defer program.deinit();
            // Lower PR -> MLIR
            const mlir_bytes = try zg.lower.lower_program_to_mlir(gpa, &program, "main", .mlir_bytecode);
            defer gpa.free(mlir_bytes);

            // Compile and serialize
            const serialized = try backend.compile_serialized(device, mlir_bytes, .mlir_bytecode, .{});
            defer gpa.free(serialized);

            try write_bytes_to_path(path.?, serialized);
            std.log.info("wrote PJRT JIT cache artifact: {d} bytes -> {s}", .{ serialized.len, path.? });
            return;
        }
        if (std.mem.eql(u8, m, "jit-cache-run")) {
            if (have_dump_pr or have_dump_mlir) {
                try print_usage();
                return error.InvalidArguments;
            }
            const path: ?[]const u8 = if (mode_args.items.len == 1) mode_args.items[0] else null;
            if (path == null) {
                try print_usage();
                return error.InvalidArguments;
            }

            const serialized = try read_bytes_from_path(gpa, path.?);
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

    const result = try exe.execute(allocator, &.{ dev_a, dev_b, dev_c });
    defer {
        if (result.device_complete_event) |ev| {
            var tmp = ev;
            tmp.deinit();
        }
        for (result.outputs) |*buf| buf.deinit();
        allocator.free(result.outputs);
    }

    if (result.outputs.len != 1) return error.UnexpectedOutputs;

    var out_host = try zg.utils.HostBuffer.init(allocator, shape_c, .f32);
    defer out_host.deinit();
    var ev = try result.outputs[0].to_host(out_host.data);
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

fn run_custom_call_negative(allocator: std.mem.Allocator, backend: *zg.backend.PjrtBackend, device: anytype, dump_pr: ?*zg.pipeline.DumpConfig, dump_mlir: ?*zg.pipeline.DumpConfig) !void {
    var program = zg.pr.Program.init(allocator);
    defer program.deinit();

    var b = try zg.pr.FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const x = try b.param_tensor(.f32, &.{ 2, 3 });
    const y = try b.custom_call("zigrad.test.missing_handler", &.{x}, x);
    const func = try b.finish(&.{y});
    try program.add_function(func);

    const lower_encoding: zg.pipeline.MlirEncoding = if (dump_mlir != null) .text else .bytecode;
    var exe = compile_program(backend, allocator, &program, device, .{
        .encoding = lower_encoding,
        .entry_name = "main",
    }, dump_pr, dump_mlir) catch |err| {
        std.log.info("OK: custom_call compile failed as expected: {s}", .{@errorName(err)});
        return;
    };
    defer exe.deinit();

    std.log.err("unexpected: custom_call compiled without a handler", .{});
    return error.UnexpectedSuccess;
}

fn run_vjp_demo(allocator: std.mem.Allocator, backend: *zg.backend.PjrtBackend, device: anytype, dump_pr: ?*zg.pipeline.DumpConfig, dump_mlir: ?*zg.pipeline.DumpConfig) !void {
    var program = try zg.frontend.build_demo_program(allocator);
    defer program.deinit();

    const fwd = program.functions[0];
    const vjp = try zg.pr.ad.vjp(allocator, &program, fwd, "main_vjp");
    try program.add_function(vjp);

    const lower_encoding: zg.pipeline.MlirEncoding = if (dump_mlir != null) .text else .bytecode;
    var exe = try compile_program(backend, allocator, &program, device, .{
        .encoding = lower_encoding,
        .entry_name = "main_vjp",
    }, dump_pr, dump_mlir);
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

    const result = try exe.execute(allocator, &.{ dev_a, dev_b, dev_c, dev_ct });
    defer {
        if (result.device_complete_event) |ev| {
            var tmp = ev;
            tmp.deinit();
        }
        for (result.outputs) |*buf| buf.deinit();
        allocator.free(result.outputs);
    }

    if (result.outputs.len != 3) return error.UnexpectedOutputs;

    var out_a = try zg.utils.HostBuffer.init(allocator, shape_a, .f32);
    defer out_a.deinit();
    var out_b = try zg.utils.HostBuffer.init(allocator, shape_b, .f32);
    defer out_b.deinit();
    var out_c = try zg.utils.HostBuffer.init(allocator, shape_c, .f32);
    defer out_c.deinit();

    var ev_a = try result.outputs[0].to_host(out_a.data);
    defer ev_a.deinit();
    var ev_b = try result.outputs[1].to_host(out_b.data);
    defer ev_b.deinit();
    var ev_c = try result.outputs[2].to_host(out_c.data);
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

fn run_train_demo(
    allocator: std.mem.Allocator,
    plugin_path: []const u8,
    dump_pr: ?*zg.pipeline.DumpConfig,
    dump_mlir: ?*zg.pipeline.DumpConfig,
    warmup_steps: usize,
) !void {
    const TensorSpec = zg.frontend.TensorSpec;

    const ParamsSpec = struct {
        w1: TensorSpec,
        b1: TensorSpec,
        w2: TensorSpec,
        b2: TensorSpec,
        w3: TensorSpec,
        b3: TensorSpec,
    };

    const BatchSpec = struct {
        x: TensorSpec,
        y: TensorSpec,
    };

    const LossFn = struct {
        fn call(params: anytype, batch: anytype) !zg.frontend.Tensor {
            const bs: usize = 64;
            const h1: usize = 128;
            const h2: usize = 64;
            const out: usize = 10;

            const z1 = try batch.x.matmul(params.w1);
            const b1b = try params.b1.broadcast_in_dim(&.{ bs, h1 }, &.{1});
            const a1 = try z1.add(b1b);

            const z2 = try a1.matmul(params.w2);
            const b2b = try params.b2.broadcast_in_dim(&.{ bs, h2 }, &.{1});
            const a2 = try z2.add(b2b);

            const z3 = try a2.matmul(params.w3);
            const b3b = try params.b3.broadcast_in_dim(&.{ bs, out }, &.{1});
            const preds = try z3.add(b3b);
            const diff = try preds.sub(batch.y);
            const sq = try diff.mul(diff);
            return try sq.reduce_sum(&.{ 0, 1 });
        }
    };


    const bs: usize = 64;
    const in_dim: usize = 784;
    const h1: usize = 128;
    const h2: usize = 64;
    const out_dim: usize = 10;

    const params_spec = ParamsSpec{
            .w1 = .{ .dtype = .f32, .dims = &.{ in_dim, h1 } },
            .b1 = .{ .dtype = .f32, .dims = &.{ h1 } },
            .w2 = .{ .dtype = .f32, .dims = &.{ h1, h2 } },
            .b2 = .{ .dtype = .f32, .dims = &.{ h2 } },
            .w3 = .{ .dtype = .f32, .dims = &.{ h2, out_dim } },
            .b3 = .{ .dtype = .f32, .dims = &.{ out_dim } },
        };
    const batch_spec = BatchSpec{
            .x = .{ .dtype = .f32, .dims = &.{ bs, in_dim } },
            .y = .{ .dtype = .f32, .dims = &.{ bs, out_dim } },
        };
    const inputs_spec = .{ params_spec, batch_spec };

    var compile_cfg = zg.frontend.CompileConfig{
        .entry_name = "train_step",
        .plugin_path = plugin_path,
        .dump_pr = if (dump_pr) |cfg| cfg.* else null,
        .dump_mlir = if (dump_mlir) |cfg| cfg.* else null,
    };
    if (compile_cfg.dump_mlir != null) {
        compile_cfg.lower.encoding = .text;
    }

    var backend_handle = try zg.frontend.init_backend(allocator, compile_cfg.plugin_path);
    defer backend_handle.deinit();

    const devices = try backend_handle.get_devices(allocator);
    defer allocator.free(devices);

    if (compile_cfg.device_index >= devices.len) return error.InvalidDeviceIndex;
    const device = &devices[compile_cfg.device_index];

    var compiled = try zg.frontend.compile_train_step(allocator, &backend_handle, device, LossFn.call, inputs_spec, 6, 1e-2, compile_cfg);
    defer compiled.deinit();

    const true_w1 = try allocator.alloc(f32, in_dim * h1);
    defer allocator.free(true_w1);
    const true_b1 = try allocator.alloc(f32, h1);
    defer allocator.free(true_b1);
    const true_w2 = try allocator.alloc(f32, h1 * h2);
    defer allocator.free(true_w2);
    const true_b2 = try allocator.alloc(f32, h2);
    defer allocator.free(true_b2);
    const true_w3 = try allocator.alloc(f32, h2 * out_dim);
    defer allocator.free(true_w3);
    const true_b3 = try allocator.alloc(f32, out_dim);
    defer allocator.free(true_b3);

    fill_pattern(true_w1, 1e-6, 0.0);
    fill_pattern(true_b1, 1e-6, 0.0);
    fill_pattern(true_w2, 1e-6, 0.0);
    fill_pattern(true_b2, 1e-6, 0.0);
    fill_pattern(true_w3, 1e-6, 0.0);
    fill_pattern(true_b3, 1e-6, 0.0);

    const shape_w1 = zg.utils.Shape{ .dims = &.{ in_dim, h1 } };
    const shape_b1 = zg.utils.Shape{ .dims = &.{ h1 } };
    const shape_w2 = zg.utils.Shape{ .dims = &.{ h1, h2 } };
    const shape_b2 = zg.utils.Shape{ .dims = &.{ h2 } };
    const shape_w3 = zg.utils.Shape{ .dims = &.{ h2, out_dim } };
    const shape_b3 = zg.utils.Shape{ .dims = &.{ out_dim } };
    const shape_x = zg.utils.Shape{ .dims = &.{ bs, in_dim } };
    const shape_y = zg.utils.Shape{ .dims = &.{ bs, out_dim } };

    var host_w1 = try zg.utils.HostBuffer.init(allocator, shape_w1, .f32);
    defer host_w1.deinit();
    var host_b1 = try zg.utils.HostBuffer.init(allocator, shape_b1, .f32);
    defer host_b1.deinit();
    var host_w2 = try zg.utils.HostBuffer.init(allocator, shape_w2, .f32);
    defer host_w2.deinit();
    var host_b2 = try zg.utils.HostBuffer.init(allocator, shape_b2, .f32);
    defer host_b2.deinit();
    var host_w3 = try zg.utils.HostBuffer.init(allocator, shape_w3, .f32);
    defer host_w3.deinit();
    var host_b3 = try zg.utils.HostBuffer.init(allocator, shape_b3, .f32);
    defer host_b3.deinit();
    var host_x = try zg.utils.HostBuffer.init(allocator, shape_x, .f32);
    defer host_x.deinit();
    var host_y = try zg.utils.HostBuffer.init(allocator, shape_y, .f32);
    defer host_y.deinit();

    fill_pattern(host_w1.as_slice(f32), 1e-7, 0.0);
    fill_pattern(host_b1.as_slice(f32), 1e-7, 0.0);
    fill_pattern(host_w2.as_slice(f32), 1e-7, 0.0);
    fill_pattern(host_b2.as_slice(f32), 1e-7, 0.0);
    fill_pattern(host_w3.as_slice(f32), 1e-7, 0.0);
    fill_pattern(host_b3.as_slice(f32), 1e-7, 0.0);
    fill_inputs(host_x.as_slice(f32));

    const scratch1 = try allocator.alloc(f32, bs * h1);
    defer allocator.free(scratch1);
    const scratch2 = try allocator.alloc(f32, bs * h2);
    defer allocator.free(scratch2);

    try fill_targets(
        &host_y,
        &host_x,
        true_w1,
        true_b1,
        true_w2,
        true_b2,
        true_w3,
        true_b3,
        scratch1,
        scratch2,
        bs,
        in_dim,
        h1,
        h2,
        out_dim,
    );

    const steps: usize = 8;
    var total_ns: u64 = 0;

    var dev_w1 = try upload_host_buffer(allocator, &backend_handle, device, &host_w1);
    defer dev_w1.deinit();
    var dev_b1 = try upload_host_buffer(allocator, &backend_handle, device, &host_b1);
    defer dev_b1.deinit();
    var dev_w2 = try upload_host_buffer(allocator, &backend_handle, device, &host_w2);
    defer dev_w2.deinit();
    var dev_b2 = try upload_host_buffer(allocator, &backend_handle, device, &host_b2);
    defer dev_b2.deinit();
    var dev_w3 = try upload_host_buffer(allocator, &backend_handle, device, &host_w3);
    defer dev_w3.deinit();
    var dev_b3 = try upload_host_buffer(allocator, &backend_handle, device, &host_b3);
    defer dev_b3.deinit();
    var dev_x = try upload_host_buffer(allocator, &backend_handle, device, &host_x);
    defer dev_x.deinit();
    var dev_y = try upload_host_buffer(allocator, &backend_handle, device, &host_y);
    defer dev_y.deinit();

    var loss_host = try zg.utils.HostBuffer.init(allocator, .{ .dims = &.{} }, .f32);
    defer loss_host.deinit();

    const output_count = 7;
    const api = dev_w1.api;

    var input_ptrs = [_]zg.backend.pjrt.RawBuffer{
        dev_w1.pjrt_buffer, dev_b1.pjrt_buffer, dev_w2.pjrt_buffer, dev_b2.pjrt_buffer,
        dev_w3.pjrt_buffer, dev_b3.pjrt_buffer, dev_x.pjrt_buffer,  dev_y.pjrt_buffer,
    };
    var output_ptrs: [output_count]zg.backend.pjrt.RawBuffer = undefined;
    const exec_opts: zg.frontend.CompiledForward.ExecuteOptions = .{
        .non_donatable_input_indices = &.{ 6, 7 },
    };

    // Check once if we're on CPU for direct memory access optimization
    const is_cpu = try dev_w1.is_on_cpu();

    var warmup: usize = 0;
    while (warmup < warmup_steps) : (warmup += 1) {
        const ev = try compiled.execute_into(&input_ptrs, &output_ptrs, exec_opts);

        var loss_buf = zg.backend.pjrt.Buffer{ .api = api, .pjrt_buffer = output_ptrs[0] };
        if (is_cpu) {
            // CPU: wait on device event (same as timed loop)
            if (ev) |e| {
                var tmp = e;
                try tmp.await_();
                tmp.deinit();
            }
        } else {
            if (ev) |e| {
                var tmp = e;
                tmp.deinit();
            }
            var loss_ev = try loss_buf.to_host(loss_host.data);
            try loss_ev.await_();
            loss_ev.deinit();
        }
        loss_buf.deinit();

        for (input_ptrs[0..6], output_ptrs[1..]) |*old, new| {
            var buf = zg.backend.pjrt.Buffer{ .api = api, .pjrt_buffer = old.* };
            buf.deinit();
            old.* = new;
        }
    }

    var step: usize = 0;
    while (step < steps) : (step += 1) {
        var timer = try std.time.Timer.start();
        const event = try compiled.execute_into(&input_ptrs, &output_ptrs, exec_opts);
        const dispatch_ns = timer.lap();

        var loss_buf = zg.backend.pjrt.Buffer{ .api = api, .pjrt_buffer = output_ptrs[0] };

        const loss: f32 = if (is_cpu) blk: {
            // CPU: wait for compute, then read directly from buffer memory
            if (event) |ev| {
                var tmp = ev;
                try tmp.await_();
                tmp.deinit();
            }
            const ptr: [*]const f32 = @ptrFromInt(try loss_buf.unsafe_pointer());
            break :blk ptr[0];
        } else blk: {
            // GPU: wait for compute via async copy to host
            if (event) |ev| {
                var tmp = ev;
                tmp.deinit();
            }
            var loss_ev = try loss_buf.to_host(loss_host.data);
            try loss_ev.await_();
            loss_ev.deinit();
            break :blk loss_host.as_slice(f32)[0];
        };
        const compute_ns = timer.lap();

        loss_buf.deinit();

        for (input_ptrs[0..6], output_ptrs[1..]) |*old, new| {
            var buf = zg.backend.pjrt.Buffer{ .api = api, .pjrt_buffer = old.* };
            buf.deinit();
            old.* = new;
        }

        const cleanup_ns = timer.lap();
        const step_ns = dispatch_ns + compute_ns + cleanup_ns;
        total_ns += step_ns;
        const step_ms = @as(f64, @floatFromInt(step_ns)) / std.time.ns_per_ms;
        const dispatch_ms = @as(f64, @floatFromInt(dispatch_ns)) / std.time.ns_per_ms;
        const compute_ms = @as(f64, @floatFromInt(compute_ns)) / std.time.ns_per_ms;
        const cleanup_ms = @as(f64, @floatFromInt(cleanup_ns)) / std.time.ns_per_ms;
        std.log.info("train-demo step {d}: loss={d:.6} dispatch={d:.3}ms compute={d:.3}ms cleanup={d:.3}ms total={d:.3}ms", .{
            step, loss, dispatch_ms, compute_ms, cleanup_ms, step_ms,
        });
    }

    const avg_ms = @as(f64, @floatFromInt(total_ns)) / std.time.ns_per_ms / @as(f64, @floatFromInt(steps));
    std.log.info("train-demo avg_step_ms={d:.3}", .{avg_ms});
    std.log.info("OK: train-demo executed", .{});
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

fn upload_host_buffer(
    allocator: std.mem.Allocator,
    backend: *zg.backend.PjrtBackend,
    device: *const zg.backend.pjrt.Device,
    buf: *zg.utils.HostBuffer,
) !zg.backend.pjrt.Buffer {
    const shape_i64 = try allocator.alloc(i64, buf.shape.dims.len);
    defer allocator.free(shape_i64);
    for (buf.shape.dims, 0..) |d, i| shape_i64[i] = @intCast(d);
    const dtype: zg.backend.pjrt.BufferType = switch (buf.dtype) {
        .f32 => .f32,
        .f64 => .f64,
        .i32 => .i32,
        .i64 => .i64,
        .u32 => .u32,
        .u64 => .u64,
    };
    return backend.buffer_from_host(device, buf.data, dtype, shape_i64);
}

fn fill_pattern(slice: []f32, scale: f32, offset: f32) void {
    for (slice, 0..) |*v, i| {
        const base = @as(f32, @floatFromInt(i % 1024));
        v.* = offset + scale * base;
    }
}

fn fill_inputs(x: []f32) void {
    for (x, 0..) |*v, i| {
        const base = @as(f32, @floatFromInt(i % 256));
        v.* = base / 255.0;
    }
}

fn fill_targets(
    host_y: *zg.utils.HostBuffer,
    host_x: *zg.utils.HostBuffer,
    w1: []const f32,
    b1: []const f32,
    w2: []const f32,
    b2: []const f32,
    w3: []const f32,
    b3: []const f32,
    scratch1: []f32,
    scratch2: []f32,
    bs: usize,
    in_dim: usize,
    h1: usize,
    h2: usize,
    out_dim: usize,
) !void {
    if (host_y.dtype != .f32 or host_x.dtype != .f32) return error.UnsupportedDType;
    if (host_x.shape.dims.len != 2 or host_y.shape.dims.len != 2) return error.ShapeMismatch;
    if (host_x.shape.dims[0] != bs or host_x.shape.dims[1] != in_dim) return error.ShapeMismatch;
    if (host_y.shape.dims[0] != bs or host_y.shape.dims[1] != out_dim) return error.ShapeMismatch;

    if (w1.len != in_dim * h1 or b1.len != h1) return error.ShapeMismatch;
    if (w2.len != h1 * h2 or b2.len != h2) return error.ShapeMismatch;
    if (w3.len != h2 * out_dim or b3.len != out_dim) return error.ShapeMismatch;
    if (scratch1.len != bs * h1 or scratch2.len != bs * h2) return error.ShapeMismatch;

    const x = host_x.as_slice(f32);
    const y = host_y.as_slice(f32);

    var n: usize = 0;
    while (n < bs) : (n += 1) {
        var j: usize = 0;
        while (j < h1) : (j += 1) {
            var acc: f32 = b1[j];
            var k: usize = 0;
            while (k < in_dim) : (k += 1) {
                acc += x[n * in_dim + k] * w1[k * h1 + j];
            }
            scratch1[n * h1 + j] = acc;
        }
    }

    n = 0;
    while (n < bs) : (n += 1) {
        var j: usize = 0;
        while (j < h2) : (j += 1) {
            var acc: f32 = b2[j];
            var k: usize = 0;
            while (k < h1) : (k += 1) {
                acc += scratch1[n * h1 + k] * w2[k * h2 + j];
            }
            scratch2[n * h2 + j] = acc;
        }
    }

    n = 0;
    while (n < bs) : (n += 1) {
        var j: usize = 0;
        while (j < out_dim) : (j += 1) {
            var acc: f32 = b3[j];
            var k: usize = 0;
            while (k < h2) : (k += 1) {
                acc += scratch2[n * h2 + k] * w3[k * out_dim + j];
            }
            y[n * out_dim + j] = acc;
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
        \\usage: zigrad [global options] [mode [args]]
        \\
        \\global options (must appear before mode):
        \\  -h, --help          show this help
        \\  --dump-pr           print PR (zxpr) to stdout (default/custom-call-neg/vjp-demo)
        \\  --dump-pr=PATH      write PR (zxpr) to PATH (default/custom-call-neg/vjp-demo)
        \\  --dump-mlir         print MLIR (text) to stdout (default/custom-call-neg/vjp-demo)
        \\  --dump-mlir=PATH    write MLIR (text) to PATH (default/custom-call-neg/vjp-demo)
        \\
        \\modes:
        \\  print-pr                     prints the PR for the demo program
        \\  custom-call-neg              expects missing custom call handler
        \\  vjp-demo                     runs the reverse-mode demo
        \\  train-demo [warmup]          runs the frontend training demo
        \\  jit-cache-save <path>        writes PJRT JIT cache artifact
        \\  jit-cache-run <path>         loads and runs PJRT JIT cache artifact
        \\
    );

    try out.flush();
}
