const std = @import("std");
const zg = @import("zigrad");

pub fn write_bytes_to_path(path: []const u8, bytes: []const u8) !void {
    var file = if (std.fs.path.isAbsolute(path))
        try std.fs.createFileAbsolute(path, .{ .truncate = true })
    else
        try std.fs.cwd().createFile(path, .{ .truncate = true });
    defer file.close();
    try file.writeAll(bytes);
}

pub fn read_bytes_from_path(allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    var file = if (std.fs.path.isAbsolute(path))
        try std.fs.openFileAbsolute(path, .{})
    else
        try std.fs.cwd().openFile(path, .{});
    defer file.close();
    return file.readToEndAlloc(allocator, std.math.maxInt(usize));
}

pub fn run_demo_executable(
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
    defer backend.deinit_buffer(&dev_a);
    var dev_b = try backend.buffer_from_host(device, host_b.data, .f32, dims_b[0..]);
    defer backend.deinit_buffer(&dev_b);
    var dev_c = try backend.buffer_from_host(device, host_c.data, .f32, dims_c[0..]);
    defer backend.deinit_buffer(&dev_c);

    const result = try backend.execute(exe, allocator, &.{ dev_a, dev_b, dev_c });
    defer {
        if (result.device_complete_event) |ev| {
            var tmp = ev;
            backend.deinit_event(&tmp);
        }
        for (result.outputs) |*buf| backend.deinit_buffer(buf);
        allocator.free(result.outputs);
    }

    if (result.outputs.len != 1) return error.UnexpectedOutputs;

    var out_host = try zg.utils.HostBuffer.init(allocator, shape_c, .f32);
    defer out_host.deinit();
    var ev = try backend.buffer_to_host(&result.outputs[0], out_host.data);
    defer backend.deinit_event(&ev);
    try backend.await_event(&ev);

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

pub fn run_custom_call_negative(allocator: std.mem.Allocator, backend: *zg.backend.PjrtBackend, device: anytype, dump_pr: ?*zg.pipeline.DumpConfig, dump_mlir: ?*zg.pipeline.DumpConfig) !void {
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
    defer backend.deinit_executable(&exe);

    std.log.err("unexpected: custom_call compiled without a handler", .{});
    return error.UnexpectedSuccess;
}

pub fn run_vjp_demo(allocator: std.mem.Allocator, backend: *zg.backend.PjrtBackend, device: anytype, dump_pr: ?*zg.pipeline.DumpConfig, dump_mlir: ?*zg.pipeline.DumpConfig) !void {
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
    defer backend.deinit_executable(&exe);

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
    defer backend.deinit_buffer(&dev_a);
    var dev_b = try backend.buffer_from_host(device, host_b.data, .f32, dims_b[0..]);
    defer backend.deinit_buffer(&dev_b);
    var dev_c = try backend.buffer_from_host(device, host_c.data, .f32, dims_c[0..]);
    defer backend.deinit_buffer(&dev_c);
    var dev_ct = try backend.buffer_from_host(device, host_ct.data, .f32, dims_c[0..]);
    defer backend.deinit_buffer(&dev_ct);

    const result = try backend.execute(&exe, allocator, &.{ dev_a, dev_b, dev_c, dev_ct });
    defer {
        if (result.device_complete_event) |ev| {
            var tmp = ev;
            backend.deinit_event(&tmp);
        }
        for (result.outputs) |*buf| backend.deinit_buffer(buf);
        allocator.free(result.outputs);
    }

    if (result.outputs.len != 3) return error.UnexpectedOutputs;

    var out_a = try zg.utils.HostBuffer.init(allocator, shape_a, .f32);
    defer out_a.deinit();
    var out_b = try zg.utils.HostBuffer.init(allocator, shape_b, .f32);
    defer out_b.deinit();
    var out_c = try zg.utils.HostBuffer.init(allocator, shape_c, .f32);
    defer out_c.deinit();

    var ev_a = try backend.buffer_to_host(&result.outputs[0], out_a.data);
    defer backend.deinit_event(&ev_a);
    var ev_b = try backend.buffer_to_host(&result.outputs[1], out_b.data);
    defer backend.deinit_event(&ev_b);
    var ev_c = try backend.buffer_to_host(&result.outputs[2], out_c.data);
    defer backend.deinit_event(&ev_c);

    try backend.await_event(&ev_a);
    try backend.await_event(&ev_b);
    try backend.await_event(&ev_c);

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

pub fn run_train_demo(
    allocator: std.mem.Allocator,
    plugin_path: []const u8,
    dump_pr: ?*zg.pipeline.DumpConfig,
    dump_mlir: ?*zg.pipeline.DumpConfig,
    warmup_steps: usize,
    steps: usize,
    quiet: bool,
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
        .b1 = .{ .dtype = .f32, .dims = &.{h1} },
        .w2 = .{ .dtype = .f32, .dims = &.{ h1, h2 } },
        .b2 = .{ .dtype = .f32, .dims = &.{h2} },
        .w3 = .{ .dtype = .f32, .dims = &.{ h2, out_dim } },
        .b3 = .{ .dtype = .f32, .dims = &.{out_dim} },
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

    const train = zg.frontend.train;
    var compiled = try train.compile_train_step(allocator, &backend_handle, device, LossFn.call, inputs_spec, 6, .{
        .optimizer = .{ .lr = 1e-2 },
        .compile = compile_cfg,
    });
    defer backend_handle.deinit_executable(&compiled.exe);

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
    const shape_b1 = zg.utils.Shape{ .dims = &.{h1} };
    const shape_w2 = zg.utils.Shape{ .dims = &.{ h1, h2 } };
    const shape_b2 = zg.utils.Shape{ .dims = &.{h2} };
    const shape_w3 = zg.utils.Shape{ .dims = &.{ h2, out_dim } };
    const shape_b3 = zg.utils.Shape{ .dims = &.{out_dim} };
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

    var total_ns: u64 = 0;

    const upload = zg.frontend.upload_host_buffer;
    const tmp_w1 = try upload(allocator, &backend_handle, device, &host_w1);
    const tmp_b1 = try upload(allocator, &backend_handle, device, &host_b1);
    const tmp_w2 = try upload(allocator, &backend_handle, device, &host_w2);
    const tmp_b2 = try upload(allocator, &backend_handle, device, &host_b2);
    const tmp_w3 = try upload(allocator, &backend_handle, device, &host_w3);
    const tmp_b3 = try upload(allocator, &backend_handle, device, &host_b3);
    const tmp_x = try upload(allocator, &backend_handle, device, &host_x);
    const tmp_y = try upload(allocator, &backend_handle, device, &host_y);

    var loss_host = try zg.utils.HostBuffer.init(allocator, .{ .dims = &.{} }, .f32);
    defer loss_host.deinit();

    var state = try train.TrainState.init(
        allocator,
        &compiled,
        &backend_handle,
        &.{ tmp_w1.pjrt_buffer, tmp_b1.pjrt_buffer, tmp_w2.pjrt_buffer, tmp_b2.pjrt_buffer, tmp_w3.pjrt_buffer, tmp_b3.pjrt_buffer },
        &.{ tmp_x.pjrt_buffer, tmp_y.pjrt_buffer },
    );
    defer state.deinit();
    // Batch buffers are not owned by TrainState; deinit them separately.
    defer {
        var bx = zg.backend.pjrt.Buffer{ .pjrt_buffer = tmp_x.pjrt_buffer };
        backend_handle.deinit_buffer(&bx);
        var by = zg.backend.pjrt.Buffer{ .pjrt_buffer = tmp_y.pjrt_buffer };
        backend_handle.deinit_buffer(&by);
    }

    const is_cpu = try backend_handle.buffer_is_on_cpu(&(zg.backend.pjrt.Buffer{ .pjrt_buffer = tmp_w1.pjrt_buffer }));

    var warmup: usize = 0;
    while (warmup < warmup_steps) : (warmup += 1) {
        var result = try state.step();
        if (result.event) |e| {
            var ev = e;
            try backend_handle.await_event(&ev);
            backend_handle.deinit_event(&ev);
        }
        if (!is_cpu and !quiet) {
            var loss_ev = try backend_handle.buffer_to_host(&result.loss_buf, loss_host.data);
            try backend_handle.await_event(&loss_ev);
            backend_handle.deinit_event(&loss_ev);
        }
        backend_handle.deinit_buffer(&result.loss_buf);
    }

    var step: usize = 0;
    while (step < steps) : (step += 1) {
        var timer = try std.time.Timer.start();
        var result = try state.step();
        const dispatch_ns = timer.lap();

        if (result.event) |e| {
            var ev = e;
            try backend_handle.await_event(&ev);
            backend_handle.deinit_event(&ev);
        }
        const wait_ns = timer.lap();

        const loss: ?f32 = if (quiet) null else if (is_cpu) blk: {
            const ptr: [*]const f32 = @ptrFromInt(try backend_handle.buffer_unsafe_pointer(&result.loss_buf));
            break :blk ptr[0];
        } else blk: {
            var loss_ev = try backend_handle.buffer_to_host(&result.loss_buf, loss_host.data);
            try backend_handle.await_event(&loss_ev);
            backend_handle.deinit_event(&loss_ev);
            break :blk loss_host.as_slice(f32)[0];
        };
        const loss_read_ns = timer.lap();

        backend_handle.deinit_buffer(&result.loss_buf);

        const cleanup_ns = timer.lap();
        const step_ns = dispatch_ns + wait_ns + loss_read_ns + cleanup_ns;
        total_ns += step_ns;
        const step_ms = @as(f64, @floatFromInt(step_ns)) / std.time.ns_per_ms;
        const dispatch_ms = @as(f64, @floatFromInt(dispatch_ns)) / std.time.ns_per_ms;
        const wait_ms = @as(f64, @floatFromInt(wait_ns)) / std.time.ns_per_ms;
        const loss_ms = @as(f64, @floatFromInt(loss_read_ns)) / std.time.ns_per_ms;
        const cleanup_ms = @as(f64, @floatFromInt(cleanup_ns)) / std.time.ns_per_ms;
        if (!quiet) {
            std.log.info("train-demo step {d}: loss={d:.6} dispatch={d:.3}ms wait={d:.3}ms loss={d:.3}ms cleanup={d:.3}ms total={d:.3}ms", .{
                step, loss.?, dispatch_ms, wait_ms, loss_ms, cleanup_ms, step_ms,
            });
        }
    }

    const avg_ms = @as(f64, @floatFromInt(total_ns)) / std.time.ns_per_ms / @as(f64, @floatFromInt(steps));
    std.log.info("train-demo avg_step_ms={d:.3}", .{avg_ms});
    std.log.info("OK: train-demo executed", .{});
}

pub fn compile_program(
    backend_handle: *zg.backend.PjrtBackend,
    allocator: std.mem.Allocator,
    program: *zg.pr.Program,
    device: *const zg.backend.pjrt.Device,
    lower_cfg: zg.lower.LowerPassConfig,
    dump_pr: ?*zg.pipeline.DumpConfig,
    dump_mlir: ?*zg.pipeline.DumpConfig,
) !zg.backend.pjrt.LoadedExecutable {
    var lower_cfg_mut = lower_cfg;

    var passes = std.ArrayList(zg.pipeline.Pass).initCapacity(allocator, 4) catch
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

    const pipeline = zg.pipeline.Pipeline{ .passes = passes.items };

    var ctx = zg.pipeline.PassContext{ .allocator = allocator };

    var artifact = try pipeline.run(.{ .pr = program }, &ctx);
    defer artifact.deinit(allocator);

    const mlir = switch (artifact) {
        .mlir => |m| m,
        else => return error.UnexpectedArtifact,
    };

    return backend_handle.compile(device, mlir.bytes, mlir.encoding == .bytecode, .{});
}

pub fn print_pr(allocator: std.mem.Allocator) !void {
    var program = try zg.frontend.build_demo_program(allocator);
    defer program.deinit();

    const fwd = program.functions[0];
    const vjp_func = try zg.pr.ad.vjp(allocator, &program, fwd, "main_vjp");

    var stdout_buffer: [8192]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    const stdout = &stdout_writer.interface;
    defer stdout.flush() catch {};

    try stdout.writeAll("=== Forward ===\n");
    try zg.pr.zxpr.emit(fwd, stdout, .auto_stdout, .{});
    try stdout.writeAll("\n=== VJP ===\n");
    try zg.pr.zxpr.emit(vjp_func, stdout, .auto_stdout, .{});
}

/// Enumerate TVM FFI global functions.
/// Writes available operations to stdout.
pub fn dump_tvm_ffi_symbols(allocator: std.mem.Allocator) !void {
    const api = zg.tvm_ffi.tvm_api;
    const Value = api.Value;

    try api.ensure_loaded(allocator);

    // Get function enumeration functor
    const factory_val = try api.call_global(allocator, "ffi.FunctionListGlobalNamesFunctor", &.{});
    defer factory_val.decref();

    const factory_handle = factory_val.as_object() orelse return error.UnexpectedTvmType;

    // Call factory() -> functor
    const functor_val = try api.call_handle(allocator, factory_handle, &.{});
    defer functor_val.decref();

    const functor_handle = functor_val.as_object() orelse return error.UnexpectedTvmType;

    // Get count: functor(-1)
    const count_val = try api.call_handle(allocator, functor_handle, &.{Value.int(-1)});
    const count: usize = @intCast(count_val.as_int() orelse return error.UnexpectedTvmType);

    // Collect and sort all function names
    var names = try std.ArrayList([]const u8).initCapacity(allocator, count);
    defer {
        for (names.items) |n| allocator.free(n);
        names.deinit(allocator);
    }

    for (0..count) |i| {
        var name_val = api.call_handle(allocator, functor_handle, &.{Value.int(@intCast(i))}) catch continue;
        const s = name_val.as_string(allocator) catch continue;
        try names.append(allocator, s);
    }

    std.mem.sort([]const u8, names.items, {}, struct {
        fn lessThan(_: void, a: []const u8, b: []const u8) bool {
            return std.mem.order(u8, a, b) == .lt;
        }
    }.lessThan);

    // Print with category headers
    var stdout_buf: [16384]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buf);
    const out = &stdout_writer.interface;
    defer out.flush() catch {};

    try out.print("# TVM FFI Global Functions (total: {d})\n#\n", .{names.items.len});

    var current_prefix: []const u8 = "";
    var category_count: usize = 0;
    for (names.items) |name| {
        const prefix = if (std.mem.indexOf(u8, name, ".")) |idx| name[0..idx] else "root";

        if (!std.mem.eql(u8, prefix, current_prefix)) {
            if (category_count > 0) try out.writeAll("\n");
            try out.print("## {s}\n", .{prefix});
            current_prefix = prefix;
            category_count += 1;
        }
        try out.print("  {s}\n", .{name});
    }
}

pub fn print_tvm_kernelize_pr(allocator: std.mem.Allocator, sweep_palettes: bool, palette: ?zg.pr.zxpr.Palette) !void {
    var program = zg.pr.Program.init(allocator);
    defer program.deinit();

    var b = try zg.frontend.Builder.init(&program, "main");
    defer b.deinit();

    const a = try b.param(.{ .dtype = .f32, .dims = &.{ 2, 3 } });
    const b_t = try b.param(.{ .dtype = .f32, .dims = &.{ 3, 2 } });
    const c = try b.param(.{ .dtype = .f32, .dims = &.{ 2, 2 } });
    const d = try b.param(.{ .dtype = .f32, .dims = &.{ 2, 2 } });

    const pre = try c.add(d);

    try b.push_region("tvm_matmul", .{ .kernelize = "tvm" });
    const dot = try a.matmul(b_t);
    try b.pop_region();

    try b.push_region("tvm_fused", .{ .kernelize = "tvm", .outline = true });
    const sum = try dot.add(pre);
    const mul = try sum.mul(c);
    try b.pop_region();

    const out = try mul.add(d);
    _ = try b.finish(&.{out});

    const func = program.functions[0];

    var stdout_buffer: [8192]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    const stdout = &stdout_writer.interface;
    defer stdout.flush() catch {};

    if (sweep_palettes) {
        const palettes = [_]zg.pr.zxpr.Palette{ .default, .alt, .nord, .gruvbox_material, .flat_dark, .catppuccin, .tokyonight };
        for (palettes) |pal| {
            try stdout.print("=== Kernelize(TVM subgraph) [{s}] ===\n", .{@tagName(pal)});
            try zg.pr.zxpr.emit(func, stdout, .auto_stdout, .{ .palette = pal });
            try stdout.writeAll("\n");
        }
    } else {
        try stdout.writeAll("=== Kernelize(TVM subgraph) ===\n");
        try zg.pr.zxpr.emit(func, stdout, .auto_stdout, .{ .palette = palette });
    }
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

/// Annotated attention pattern demo.
///
/// Builds simplified attention compute: Q @ K^T -> scale -> softmax -> @ V
/// with region annotations to visualize what would be lowered.
pub fn print_tvm_attention_pr(allocator: std.mem.Allocator, sweep_palettes: bool, palette: ?zg.pr.zxpr.Palette) !void {
    var program = zg.pr.Program.init(allocator);
    defer program.deinit();

    var b = try zg.frontend.Builder.init(&program, "attention");
    defer b.deinit();

    // simplified attention: [B, S, D] shapes
    // Q, K, V: [batch=2, seq=4, head_dim=64]
    const batch: i64 = 2;
    const seq: i64 = 4;
    const head_dim: i64 = 64;

    const q = try b.param(.{ .dtype = .f32, .dims = &.{ batch, seq, head_dim } });
    const k = try b.param(.{ .dtype = .f32, .dims = &.{ batch, seq, head_dim } });
    const v = try b.param(.{ .dtype = .f32, .dims = &.{ batch, seq, head_dim } });

    // entire attention block as a single TVM-kernelizable region
    try b.push_region("attention", .{ .kernelize = "tvm" });

    // attention scores: Q @ K^T  ->  [B, S, S]
    const scores = try q.dot_general(k, .{
        .lhs_batch_dims = &.{0},
        .rhs_batch_dims = &.{0},
        .lhs_contracting_dims = &.{2},
        .rhs_contracting_dims = &.{2},
    });

    // scale scores
    const scale_val = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));
    const scale = try b.scalar_literal(zg.pr.ops.types.scalar_literal(.f32, scale_val));
    const scale_broadcast = try scale.broadcast_in_dim(scores.tensor.shape.dims, &.{});
    const scaled = try scores.mul(scale_broadcast);

    // softmax over last dim [S]
    const rank = scaled.tensor.shape.dims.len;
    const axis: i64 = @intCast(rank - 1);
    const max_val = try scaled.reduce_max(&.{axis});

    // broadcast max back to full shape for stability
    const max_broadcast = try max_val.broadcast_in_dim(scaled.tensor.shape.dims, &.{ 0, 1 });
    const shifted = try scaled.sub(max_broadcast);
    const exp_vals = try shifted.exp();
    const sum_exp = try exp_vals.reduce_sum(&.{axis});
    const sum_broadcast = try sum_exp.broadcast_in_dim(exp_vals.tensor.shape.dims, &.{ 0, 1 });
    const attn_weights = try exp_vals.div(sum_broadcast);

    // output: attn_weights @ V  ->  [B, S, D]
    const out = try attn_weights.dot_general(v, .{
        .lhs_batch_dims = &.{0},
        .rhs_batch_dims = &.{0},
        .lhs_contracting_dims = &.{2},
        .rhs_contracting_dims = &.{1},
    });

    try b.pop_region();

    _ = try b.finish(&.{out});

    const func = program.functions[0];

    var stdout_buffer: [16384]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    const stdout = &stdout_writer.interface;
    defer stdout.flush() catch @panic("Flush failed");

    if (sweep_palettes) {
        const palettes = [_]zg.pr.zxpr.Palette{ .default, .alt, .nord, .gruvbox_material, .flat_dark, .catppuccin, .tokyonight };
        for (palettes) |pal| {
            try stdout.print("\n==== Palette: {s} ====\n", .{@tagName(pal)});
            try zg.pr.zxpr.emit(func, stdout, .auto_stdout, .{ .palette = pal });
        }
    } else {
        const pal = palette orelse .default;
        try zg.pr.zxpr.emit(func, stdout, .auto_stdout, .{ .palette = pal });
    }
}
