const std = @import("std");
const zg = @import("zigrad");

const log = std.log.scoped(.@"zg/demos");


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

pub fn run_custom_call_negative(allocator: std.mem.Allocator, backend: *zg.backend.PjrtBackend, device: anytype, dump_pr: ?*zg.pipeline.DumpConfig, dump_mlir: ?*zg.pipeline.DumpConfig, dump_optimized: ?*zg.pipeline.DumpConfig) !void {
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
    }, dump_pr, dump_mlir, dump_optimized, null) catch |err| {
        std.log.info("OK: custom_call compile failed as expected: {s}", .{@errorName(err)});
        return;
    };
    defer backend.deinit_executable(&exe);

    std.log.err("unexpected: custom_call compiled without a handler", .{});
    return error.UnexpectedSuccess;
}

pub fn run_vjp_demo(allocator: std.mem.Allocator, backend: *zg.backend.PjrtBackend, device: anytype, dump_pr: ?*zg.pipeline.DumpConfig, dump_mlir: ?*zg.pipeline.DumpConfig, dump_optimized: ?*zg.pipeline.DumpConfig) !void {
    var program = try zg.frontend.build_demo_program(allocator);
    defer program.deinit();

    const fwd = program.functions[0];
    const vjp = try zg.pr.ad.vjp(allocator, &program, fwd, "main_vjp");
    try program.add_function(vjp);

    const lower_encoding: zg.pipeline.MlirEncoding = if (dump_mlir != null) .text else .bytecode;
    var exe = try compile_program(backend, allocator, &program, device, .{
        .encoding = lower_encoding,
        .entry_name = "main_vjp",
    }, dump_pr, dump_mlir, dump_optimized, null);
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
    backend_handle: *zg.backend.PjrtBackend,
    device: *const zg.backend.pjrt.Device,
    dump_pr: ?*zg.pipeline.DumpConfig,
    dump_mlir: ?*zg.pipeline.DumpConfig,
    dump_optimized: ?*zg.pipeline.DumpConfig,
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
        .dump_pr = if (dump_pr) |cfg| cfg.* else null,
        .dump_mlir = if (dump_mlir) |cfg| cfg.* else null,
        .dump_optimized = if (dump_optimized) |cfg| cfg.* else null,
    };
    if (compile_cfg.dump_mlir != null) {
        compile_cfg.lower.encoding = .text;
    }

    const train = zg.frontend.train;
    var compiled = try train.compile_train_step(allocator, backend_handle, device, LossFn.call, inputs_spec, 6, .{
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
    const tmp_w1 = try upload(allocator, backend_handle, device, &host_w1);
    const tmp_b1 = try upload(allocator, backend_handle, device, &host_b1);
    const tmp_w2 = try upload(allocator, backend_handle, device, &host_w2);
    const tmp_b2 = try upload(allocator, backend_handle, device, &host_b2);
    const tmp_w3 = try upload(allocator, backend_handle, device, &host_w3);
    const tmp_b3 = try upload(allocator, backend_handle, device, &host_b3);
    const tmp_x = try upload(allocator, backend_handle, device, &host_x);
    const tmp_y = try upload(allocator, backend_handle, device, &host_y);

    var loss_host = try zg.utils.HostBuffer.init(allocator, .{ .dims = &.{} }, .f32);
    defer loss_host.deinit();

    var state = try train.TrainState.init(
        allocator,
        &compiled,
        backend_handle,
        &.{ tmp_w1.pjrt_buffer, tmp_b1.pjrt_buffer, tmp_w2.pjrt_buffer, tmp_b2.pjrt_buffer, tmp_w3.pjrt_buffer, tmp_b3.pjrt_buffer },
        &.{ tmp_x.pjrt_buffer, tmp_y.pjrt_buffer },
    );
    defer state.deinit();
    // Batch buffers are not owned by TrainState, deinit them separately
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

/// End-to-end kernel provider demo.
///
/// Compiles and executes a small matmul program through the kernelization
///  pipeline. Accepts one or more providers, each gets its own kernelized
///  region in the program.
///
/// TVM requires `ZG_EXTERNAL_SDK_ROOT` and XLA typed-FFI support, Mirage
///  requires the Mirage shared library. Both can be enabled individually 
///  or simultaneously.
pub fn run_kernel_provider_demo(
    allocator: std.mem.Allocator,
    backend: *zg.backend.PjrtBackend,
    device: *const zg.backend.pjrt.Device,
    dump_pr: ?*zg.pipeline.DumpConfig,
    dump_mlir: ?*zg.pipeline.DumpConfig,
    dump_optimized: ?*zg.pipeline.DumpConfig,
    provider_kinds: []const KernelProviderDemoKind,
    pipeline_kind: KernelProviderDemoPipeline,
) !void {
    const lane: zg.lower.KernelizationLane = switch (pipeline_kind) {
        .pr => .pr,
        .mlir => .mlir,
    };

    var registry = zg.kernel.KernelRegistry.init(allocator);
    defer registry.deinit();
    var package = zg.kernel.KernelPackage.init(allocator);
    defer package.deinit();
    try backend.register_kernel_dispatcher();

    // --- TVM setup (requires TVM headers in SDK) ---
    var tvm_dispatch: if (zg.build_options.has_tvm) zg.tvm.dispatch.TvmDispatchState else void = undefined;
    var tvm_impl: if (zg.build_options.has_tvm) zg.tvm.provider.TvmProvider else void = undefined;
    var has_tvm = false;

    if (kind_requested(provider_kinds, .tvm)) {
        if (comptime !zg.build_options.has_tvm) {
            log.err("tvm provider requested but binary was built without TVM support (headers not found in SDK)", .{});
            return error.TvmUnavailable;
        }
        try zg.tvm.ffi.ensure_loaded(allocator, .{});
        try backend.require_typed_ffi();
        const target_kind: zg.tvm.tir.TargetKind = if (backend.is_cuda()) .cuda else .cpu;
        tvm_dispatch = zg.tvm.dispatch.TvmDispatchState.init(allocator);
        tvm_impl = .{
            .allocator = allocator,
            .target_kind = target_kind,
            .work_dir = "artifacts/tvm_cache",
            .max_trials = 8,
            .trials_per_iter = 4,
            .dispatch_state = &tvm_dispatch,
        };
        has_tvm = true;
    }
    defer if (zg.build_options.has_tvm and has_tvm) tvm_dispatch.deinit();

    // --- Mirage setup (requires mirage headers in SDK) ---
    var mirage_dispatch: if (zg.build_options.has_mirage) zg.mirage.dispatch.MirageDispatchState else void = undefined;
    var mirage_impl: if (zg.build_options.has_mirage) zg.mirage.provider.MirageProvider else void = undefined;
    var has_mirage = false;

    if (kind_requested(provider_kinds, .mirage)) {
        if (comptime !zg.build_options.has_mirage) {
            log.err("mirage provider requested but binary was built without mirage support (headers not found in SDK)", .{});
            return error.MirageUnavailable;
        }
        mirage_dispatch = try zg.mirage.dispatch.MirageDispatchState.init(allocator);
        mirage_impl = .{
            .allocator = allocator,
            .dispatch_state = &mirage_dispatch,
        };
        has_mirage = true;
    }
    defer if (zg.build_options.has_mirage and has_mirage) mirage_dispatch.deinit();

    // Collect providers in requested order.
    var providers_buf: [2]zg.kernel.KernelProvider = undefined;
    var n_providers: usize = 0;
    for (provider_kinds) |kind| switch (kind) {
        .tvm => if (zg.build_options.has_tvm and has_tvm) {
            providers_buf[n_providers] = tvm_impl.kernel_provider();
            n_providers += 1;
        },
        .mirage => if (zg.build_options.has_mirage and has_mirage) {
            providers_buf[n_providers] = mirage_impl.kernel_provider();
            n_providers += 1;
        },
    };
    const providers = providers_buf[0..n_providers];

    // Build provider name strings from kinds.
    var pnames_buf: [2][]const u8 = undefined;
    for (provider_kinds, 0..) |kind, i| pnames_buf[i] = @tagName(kind);
    const provider_names = pnames_buf[0..provider_kinds.len];

    var program = try build_kernelized_demo_program(allocator, provider_names);
    defer program.deinit();

    const lower_encoding: zg.pipeline.MlirEncoding = if (dump_mlir != null) .text else .bytecode;
    var exe = try compile_program(backend, allocator, &program, device, .{
        .encoding = lower_encoding,
        .entry_name = "main",
        .kernelization_lane = lane,
    }, dump_pr, dump_mlir, dump_optimized, .{
        .registry = &registry,
        .package = &package,
        .providers = providers,
        .lane = lane,
    });
    defer backend.deinit_executable(&exe);

    return run_kernel_provider_demo_executable(allocator, backend, device, &exe, provider_kinds.len);
}

fn kind_requested(kinds: []const KernelProviderDemoKind, target: KernelProviderDemoKind) bool {
    for (kinds) |k| if (k == target) return true;
    return false;
}

/// Execute the kernel provider demo program and verify results.
fn run_kernel_provider_demo_executable(
    allocator: std.mem.Allocator,
    backend: *zg.backend.PjrtBackend,
    device: *const zg.backend.pjrt.Device,
    exe: *zg.backend.pjrt.LoadedExecutable,
    n_providers: usize,
) !void {
    const A = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    const B = [_]f32{ 7.0, 8.0, 9.0, 10.0, 11.0, 12.0 };
    const C = [_]f32{ 2.0, 2.0, 2.0, 2.0 };
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

    // dot(A, B) = [[58, 64], [139, 154]]
    // (n*dot + C) * C = [[116n+4, 128n+4], [278n+4, 308n+4]]
    const n = @as(f32, @floatFromInt(n_providers));
    const expected = [_]f32{
        116.0 * n + 4.0, 128.0 * n + 4.0,
        278.0 * n + 4.0, 308.0 * n + 4.0,
    };
    const out = out_host.as_slice(f32)[0..4];
    for (out, 0..) |v, i| {
        const diff = @abs(v - expected[i]);
        if (diff > 1e-4) {
            std.log.err("mismatch[{d}]: got {d}, expected {d}", .{ i, v, expected[i] });
            return error.NumericalMismatch;
        }
    }
    std.log.info("OK: kernelized demo output matches expected ({d} provider(s))", .{n_providers});
}

pub const KernelProviderDemoKind = enum {
    tvm,
    mirage,
};

pub const KernelProviderDemoPipeline = enum {
    pr,
    mlir,
};

pub fn compile_program(
    backend_handle: *zg.backend.PjrtBackend,
    allocator: std.mem.Allocator,
    program: *zg.pr.Program,
    device: *const zg.backend.pjrt.Device,
    lower_cfg: zg.lower.LowerPassConfig,
    dump_pr: ?*zg.pipeline.DumpConfig,
    dump_mlir: ?*zg.pipeline.DumpConfig,
    dump_optimized: ?*zg.pipeline.DumpConfig,
    kernelize_cfg: ?KernelizeConfig,
) !zg.backend.pjrt.LoadedExecutable {
    var lower_cfg_mut = lower_cfg;

    var passes = std.ArrayList(zg.pipeline.Pass).initCapacity(allocator, 6) catch
        return error.OutOfMemory;
    defer passes.deinit(allocator);

    var dump_pr_local: ?zg.pipeline.DumpConfig = null;
    if (dump_pr) |cfg| {
        dump_pr_local = cfg.*;
        dump_pr_local.?.entry_name = dump_pr_local.?.entry_name orelse lower_cfg.entry_name;
        try passes.append(allocator, zg.pipeline.dump_pr_pass_with_config(&dump_pr_local.?));
    }

    var kernelize_state: ?zg.pipeline.KernelizePass = null;
    var mlir_materialize_state: ?zg.lower.mlir.MlirKernelMaterializePass = null;
    if (kernelize_cfg) |cfg| {
        lower_cfg_mut.kernelization_lane = cfg.lane;

        if (cfg.lane == .mlir) {
            if (cfg.package == null) return error.ValidationFailed;
        }

        if (cfg.lane == .pr) {
            const package_for_kernelize = if (cfg.lane == .pr) cfg.package else null;
            kernelize_state = .{
                .registry = cfg.registry,
                .package = package_for_kernelize,
                .providers = cfg.providers,
                .rewrite_regions = cfg.lane == .pr,
                .target_name_mode = .region_name,
                .dump_kernels = cfg.dump_kernels,
            };
            try passes.append(allocator, kernelize_state.?.pass());
        }

        if (cfg.lane == .mlir) {
            if (cfg.package) |pkg| {
                mlir_materialize_state = .{
                    .registry = cfg.registry,
                    .package = pkg,
                    .providers = cfg.providers,
                    .dump_kernels = cfg.dump_kernels,
                };
            }
        }
    }

    try passes.append(allocator, zg.lower.validate_pass);
    try passes.append(allocator, zg.lower.lower_pass_with_config(&lower_cfg_mut));

    // MLIR-stage passes: explicit pipeline ordering.
    if (kernelize_cfg) |cfg| {
        if (cfg.lane == .mlir) {
            try passes.append(allocator, zg.lower.mlir.MlirSelectPass.pass());
            if (mlir_materialize_state) |*state| {
                try passes.append(allocator, state.pass());
            }
        }
    }
    try passes.append(allocator, zg.lower.mlir.stablehlo.MlirLegalizePass.pass());

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

    if (kernelize_cfg) |cfg| {
        for (cfg.providers) |provider| provider.finalize();
    }

    const mlir = switch (artifact) {
        .mlir => |m| m,
        else => return error.UnexpectedArtifact,
    };

    var compile_opts: zg.backend.pjrt.CompileOptions = .{};
    if (compile_opts.kernel_package == null) {
        compile_opts.kernel_package = if (kernelize_cfg) |cfg|
            cfg.package orelse mlir.kernel_package
        else
            mlir.kernel_package;
    }
    if (compile_opts.kernel_registry == null) {
        compile_opts.kernel_registry = if (kernelize_cfg) |cfg| cfg.registry else null;
    }
    var exe = try backend_handle.compile(device, mlir.bytes, mlir.encoding == .bytecode, compile_opts);

    if (dump_optimized) |cfg| {
        const maybe_opt = exe.get_optimized_program(backend_handle.api, allocator) catch |err| {
            log.err("get_optimized_program failed: {s}", .{@errorName(err)});
            return exe;
        };
        if (maybe_opt) |opt_const| {
            var opt = opt_const;
            defer opt.deinit(allocator);
            zg.pipeline.dump_optimized_program(cfg, opt.code, opt.format, allocator) catch |err| {
                log.err("dump-optimized failed: {s}", .{@errorName(err)});
            };
        }
    }

    return exe;
}

pub const KernelizeConfig = struct {
    /// Destination registry where kernel artifacts are stored by kernelize pass.
    registry: *zg.kernel.KernelRegistry,
    /// Optional executable-scoped package populated by kernel id.
    package: ?*zg.kernel.KernelPackage = null,
    /// Kernel providers available to kernelize pass (e.g. TVM).
    providers: []const zg.kernel.KernelProvider,

    /// Select the kernelization lane for this compile.
    lane: zg.lower.KernelizationLane = .pr,
    /// Print a summary table of kernelized regions after the pass.
    dump_kernels: bool = false,
};

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
    try zg.pr.zxpr.emit(fwd, stdout, zg.pr.zxpr.style.config(.auto_stdout, .{}));
    try stdout.writeAll("\n=== VJP ===\n");
    try zg.pr.zxpr.emit(vjp_func, stdout, zg.pr.zxpr.style.config(.auto_stdout, .{}));
}

/// Enumerate TVM FFI global functions.
/// Writes available operations to stdout.
pub fn dump_tvm_ffi_symbols(allocator: std.mem.Allocator) !void {
    if (comptime !zg.build_options.has_tvm) {
        log.err("TVM FFI symbol dump requires TVM support (headers not found in SDK)", .{});
        return error.TvmUnavailable;
    }
    const tvm_api = zg.tvm.ffi;

    try tvm_api.ensure_loaded(allocator, .{ .load_compiler = false });

    const names = try tvm_api.list_global_names(allocator);
    defer {
        for (names) |n| allocator.free(n);
        allocator.free(names);
    }

    // Print with category headers
    var stdout_buf: [16384]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buf);
    const out = &stdout_writer.interface;
    defer out.flush() catch {};

    try out.print("# TVM FFI Global Functions (total: {d})\n#\n", .{names.len});

    var current_prefix: []const u8 = "";
    var category_count: usize = 0;
    for (names) |name| {
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

pub fn print_tvm_kernelize_pr(allocator: std.mem.Allocator, sweep_palettes: bool, palette: ?zg.pr.zxpr.style.Palette) !void {
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
        const palettes = [_]zg.pr.zxpr.style.Palette{ .default, .alt, .nord, .gruvbox_material, .flat_dark, .catppuccin, .tokyonight };
        for (palettes) |pal| {
            try stdout.print("=== Kernelize(TVM subgraph) [{s}] ===\n", .{@tagName(pal)});
            try zg.pr.zxpr.emit(func, stdout, zg.pr.zxpr.style.config(.auto_stdout, .{ .palette = pal }));
            try stdout.writeAll("\n");
        }
    } else {
        try stdout.writeAll("=== Kernelize(TVM subgraph) ===\n");
        try zg.pr.zxpr.emit(func, stdout, zg.pr.zxpr.style.config(.auto_stdout, .{ .palette = palette }));
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

/// Builds a program for kernel provider demo.
///
/// Inputs: 
///   a (2x3), b (3x2), c (2x2).
///   One `dot(a,b)` kernelized region per provider, fold-summed into `sum`.
/// Output:
///   (sum + c) * c
/// Expected:
///   [[116n+4, 128n+4], [278n+4, 308n+4]] where n = provider count
fn build_kernelized_demo_program(allocator: std.mem.Allocator, provider_names: []const []const u8) !zg.pr.Program {
    var program = zg.pr.Program.init(allocator);
    errdefer program.deinit();

    var b = try zg.pr.FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const a_id = try b.param_tensor(.f32, &.{ 2, 3 });
    const b_id = try b.param_tensor(.f32, &.{ 3, 2 });
    const c_id = try b.param_tensor(.f32, &.{ 2, 2 });

    // Stack-allocate name buffers, names must outlive the builder (used within this function).
    var region_name_bufs: [2][64]u8 = undefined;

    const first_name = try std.fmt.bufPrint(&region_name_bufs[0], "{s}_region_0", .{provider_names[0]});
    try b.push_region(first_name, .{ .kernelize = provider_names[0] });
    var acc_id = try b.dot(a_id, b_id);
    try b.pop_region();

    for (provider_names[1..], 1..) |pname, i| {
        const rn = try std.fmt.bufPrint(&region_name_bufs[i], "{s}_region_{d}", .{ pname, i });
        try b.push_region(rn, .{ .kernelize = pname });
        const dot_id = try b.dot(a_id, b_id);
        try b.pop_region();
        acc_id = try b.add(acc_id, dot_id);
    }

    const add_id = try b.add(acc_id, c_id);
    const out_id = try b.multiply(add_id, c_id);

    const func = try b.finish(&.{out_id});
    try program.add_function(func);
    return program;
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
pub fn print_tvm_attention_pr(allocator: std.mem.Allocator, sweep_palettes: bool, palette: ?zg.pr.zxpr.style.Palette) !void {
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
        const palettes = [_]zg.pr.zxpr.style.Palette{ .default, .alt, .nord, .gruvbox_material, .flat_dark, .catppuccin, .tokyonight };
        for (palettes) |pal| {
            try stdout.print("\n==== Palette: {s} ====\n", .{@tagName(pal)});
            try zg.pr.zxpr.emit(func, stdout, zg.pr.zxpr.style.config(.auto_stdout, .{ .palette = pal }));
        }
    } else {
        const pal = palette orelse .default;
        try zg.pr.zxpr.emit(func, stdout, zg.pr.zxpr.style.config(.auto_stdout, .{ .palette = pal }));
    }
}
