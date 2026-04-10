const std = @import("std");
const zg = @import("zigrad");

const Tensor = zg.Tensor;
const log = std.log.scoped(.@"zg/demos");

// TODO: need to resolve and remove this
fn deinit_tensor(t: *Tensor) void {
    t.*.deinit();
}

/// Demo program: out = (dot(A, B) + C) * C where A: 2x3, B: 3x2, C: 2x2.
pub fn build_demo_program(allocator: std.mem.Allocator) !zg.pr.Program {
    var program = zg.pr.Program.init(allocator);
    errdefer program.deinit();

    var b = try zg.pr.FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const a_id = try b.param_tensor(.f32, &.{ 2, 3 });
    const b_id = try b.param_tensor(.f32, &.{ 3, 2 });
    const c_id = try b.param_tensor(.f32, &.{ 2, 2 });

    const dot_id = try b.dot(a_id, b_id);
    const add_id = try b.add(dot_id, c_id);
    const out_id = try b.multiply(add_id, c_id);

    const func = try b.finish(&.{out_id});
    try program.add_function(func);

    return program;
}

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
    b: *zg.Backend,
    device: zg.Backend.Device,
    exe: zg.Backend.Executable,
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

    // TODO: in the init path, do we really want to require this sliceAsBytes pattern or just use comptime?
    const host_a = try Tensor.host(.f32, &.{ 2, 3 }, .{ .borrow = std.mem.sliceAsBytes(&A) });
    defer host_a.deinit();
    const host_b = try Tensor.host(.f32, &.{ 3, 2 }, .{ .borrow = std.mem.sliceAsBytes(&B) });
    defer host_b.deinit();
    const host_c = try Tensor.host(.f32, &.{ 2, 2 }, .{ .borrow = std.mem.sliceAsBytes(&C) });
    defer host_c.deinit();

    const dev_a = try b.buffer_from_host(device, host_a.host_data(), .f32, host_a.dims());
    defer b.deinit_buffer(dev_a);
    const dev_b = try b.buffer_from_host(device, host_b.host_data(), .f32, host_b.dims());
    defer b.deinit_buffer(dev_b);
    const dev_c = try b.buffer_from_host(device, host_c.host_data(), .f32, host_c.dims());
    defer b.deinit_buffer(dev_c);

    const result = try b.execute(exe, allocator, &.{ dev_a, dev_b, dev_c }, .{});
    defer {
        if (result.event) |ev| b.deinit_event(ev);
        for (result.outputs) |buf| b.deinit_buffer(buf);
        allocator.free(result.outputs);
    }

    if (result.outputs.len != 1) return error.UnexpectedOutputs;

    var out_host = try Tensor.host(.f32, &.{ 2, 2 }, .{ .alloc = allocator });
    defer out_host.deinit();
    if (try b.buffer_to_host(result.outputs[0], out_host.host_data_mut())) |ev| {
        defer b.deinit_event(ev);
        try b.await_event(ev);
    }

    const out = out_host.as_const_slice(f32);
    std.debug.assert(out.len == 4);
    const expected = [_]f32{
        120.0, 132.0,
        282.0, 312.0,
    };

    for (out, 0..) |v, i| {
        const diff = @abs(v - expected[i]);
        if (diff > 1e-4) {
            log.err("mismatch[{d}]: got {d}, expected {d}", .{ i, v, expected[i] });
            return error.NumericalMismatch;
        }
    }

    log.info("OK: demo output matches expected", .{});
}

pub fn run_custom_call_negative(allocator: std.mem.Allocator, backend: *zg.Backend, device: zg.Backend.Device, dump_pr: ?*zg.pipeline.DumpConfig, dump_mlir: ?*zg.pipeline.DumpConfig, dump_optimized: ?*zg.pipeline.DumpConfig) !void {
    var program = zg.pr.Program.init(allocator);
    defer program.deinit();

    var b = try zg.pr.FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const x = try b.param_tensor(.f32, &.{ 2, 3 });
    const y = try b.custom_call("zigrad.test.missing_handler", &.{x}, x);
    const func = try b.finish(&.{y});
    try program.add_function(func);

    const lower_encoding: zg.pipeline.MlirEncoding = if (dump_mlir != null) .text else .bytecode;
    const exe = zg.frontend.compile_program(backend, allocator, &program, device, "main", .{
        .lower = .{ .encoding = lower_encoding },
        .dump_pr = if (dump_pr) |cfg| cfg.* else null,
        .dump_mlir = if (dump_mlir) |cfg| cfg.* else null,
        .dump_optimized = if (dump_optimized) |cfg| cfg.* else null,
    }) catch |err| {
        log.info("OK: custom_call compile failed as expected: {s}", .{@errorName(err)});
        return;
    };
    defer backend.deinit_executable(exe);

    log.err("unexpected: custom_call compiled without a handler", .{});
    return error.UnexpectedSuccess;
}

pub fn run_vjp_demo(allocator: std.mem.Allocator, backend: *zg.Backend, device: zg.Backend.Device, dump_pr: ?*zg.pipeline.DumpConfig, dump_mlir: ?*zg.pipeline.DumpConfig, dump_optimized: ?*zg.pipeline.DumpConfig) !void {
    var program = try build_demo_program(allocator);
    defer program.deinit();

    const fwd = program.functions[0];
    const vjp = try zg.pr.ad.vjp(allocator, &program, fwd, "main_vjp");
    try program.add_function(vjp);

    const lower_encoding: zg.pipeline.MlirEncoding = if (dump_mlir != null) .text else .bytecode;
    const exe = try zg.frontend.compile_program(backend, allocator, &program, device, "main_vjp", .{
        .lower = .{ .encoding = lower_encoding },
        .dump_pr = if (dump_pr) |cfg| cfg.* else null,
        .dump_mlir = if (dump_mlir) |cfg| cfg.* else null,
        .dump_optimized = if (dump_optimized) |cfg| cfg.* else null,
    });
    defer backend.deinit_executable(exe);

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

    const host_a = try Tensor.host(.f32, &.{ 2, 3 }, .{ .borrow = std.mem.sliceAsBytes(&A) });
    defer host_a.deinit();
    const host_b = try Tensor.host(.f32, &.{ 3, 2 }, .{ .borrow = std.mem.sliceAsBytes(&B) });
    defer host_b.deinit();
    const host_c = try Tensor.host(.f32, &.{ 2, 2 }, .{ .borrow = std.mem.sliceAsBytes(&C) });
    defer host_c.deinit();
    const host_ct = try Tensor.host(.f32, &.{ 2, 2 }, .{ .borrow = std.mem.sliceAsBytes(&CtOut) });
    defer host_ct.deinit();

    const dev_a = try backend.buffer_from_host(device, host_a.host_data(), .f32, host_a.dims());
    defer backend.deinit_buffer(dev_a);
    const dev_b = try backend.buffer_from_host(device, host_b.host_data(), .f32, host_b.dims());
    defer backend.deinit_buffer(dev_b);
    const dev_c = try backend.buffer_from_host(device, host_c.host_data(), .f32, host_c.dims());
    defer backend.deinit_buffer(dev_c);
    const dev_ct = try backend.buffer_from_host(device, host_ct.host_data(), .f32, host_ct.dims());
    defer backend.deinit_buffer(dev_ct);

    const result = try backend.execute(exe, allocator, &.{ dev_a, dev_b, dev_c, dev_ct }, .{});
    defer {
        if (result.event) |ev| backend.deinit_event(ev);
        for (result.outputs) |buf| backend.deinit_buffer(buf);
        allocator.free(result.outputs);
    }

    if (result.outputs.len != 3) return error.UnexpectedOutputs;

    var out_a = try Tensor.host(.f32, &.{ 2, 3 }, .{ .alloc = allocator });
    defer out_a.deinit();
    var out_b = try Tensor.host(.f32, &.{ 3, 2 }, .{ .alloc = allocator });
    defer out_b.deinit();
    var out_c = try Tensor.host(.f32, &.{ 2, 2 }, .{ .alloc = allocator });
    defer out_c.deinit();

    const ev_a = try backend.buffer_to_host(result.outputs[0], out_a.host_data_mut());
    defer if (ev_a) |ev| backend.deinit_event(ev);
    const ev_b = try backend.buffer_to_host(result.outputs[1], out_b.host_data_mut());
    defer if (ev_b) |ev| backend.deinit_event(ev);
    const ev_c = try backend.buffer_to_host(result.outputs[2], out_c.host_data_mut());
    defer if (ev_c) |ev| backend.deinit_event(ev);

    if (ev_a) |ev| try backend.await_event(ev);
    if (ev_b) |ev| try backend.await_event(ev);
    if (ev_c) |ev| try backend.await_event(ev);

    const got_a = out_a.as_const_slice(f32);
    const got_b = out_b.as_const_slice(f32);
    const got_c = out_c.as_const_slice(f32);
    std.debug.assert(got_a.len == 6);
    std.debug.assert(got_b.len == 6);
    std.debug.assert(got_c.len == 4);

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

    try expect_all_close("dA", got_a, &expected_a, 1e-4);
    try expect_all_close("dB", got_b, &expected_b, 1e-4);
    try expect_all_close("dC", got_c, &expected_c, 1e-4);

    log.info("OK: gradients match expected", .{});
}

pub fn run_train_demo(
    allocator: std.mem.Allocator,
    backend: *zg.Backend,
    device: zg.Backend.Device,
    dump_pr: ?*zg.pipeline.DumpConfig,
    dump_mlir: ?*zg.pipeline.DumpConfig,
    dump_optimized: ?*zg.pipeline.DumpConfig,
    warmup_steps: usize,
    steps: usize,
    quiet: bool,
) !void {
    const ParamsSpec = struct {
        w1: Tensor,
        b1: Tensor,
        w2: Tensor,
        b2: Tensor,
        w3: Tensor,
        b3: Tensor,
    };

    const BatchSpec = struct {
        x: Tensor,
        y: Tensor,
    };

    const Fns = struct {
        fn loss(params: ParamsSpec, batch: BatchSpec) !Tensor {
            const bs_: i64 = 64;
            const h1_: i64 = 128;
            const h2_: i64 = 64;
            const out_: i64 = 10;

            const z1 = try batch.x.matmul(params.w1);
            const b1b = try params.b1.broadcast_in_dim(&.{ bs_, h1_ }, &.{1});
            const a1 = try z1.add(b1b);

            const z2 = try a1.matmul(params.w2);
            const b2b = try params.b2.broadcast_in_dim(&.{ bs_, h2_ }, &.{1});
            const a2 = try z2.add(b2b);

            const z3 = try a2.matmul(params.w3);
            const b3b = try params.b3.broadcast_in_dim(&.{ bs_, out_ }, &.{1});
            const preds = try z3.add(b3b);
            const diff = try preds.sub(batch.y);
            const sq = try diff.mul(diff);
            return try sq.reduce_sum(&.{ 0, 1 });
        }

        fn train_step(params: ParamsSpec, batch: BatchSpec) !struct { loss_val: Tensor, updated: ParamsSpec } {
            var vg = try zg.frontend.transforms.value_and_grad(loss, .{ params, batch });
            defer vg.deinit();
            var params_tree = try zg.utils.Tree(Tensor).from(vg.grads.allocator, params);
            defer params_tree.deinit();
            const optim = zg.frontend.optim.SGD{ .lr = 1e-2 };
            var updated = try params_tree.map2(Tensor, &vg.grads, Tensor, optim, zg.frontend.optim.SGD.update);
            defer updated.deinit();
            return .{
                .loss_val = vg.value,
                .updated = try updated.extract(ParamsSpec),
            };
        }
    };

    const bs: i64 = 64;
    const in_dim: i64 = 784;
    const h1: i64 = 128;
    const h2: i64 = 64;
    const out_dim: i64 = 10;

    const params_spec = ParamsSpec{
        .w1 = Tensor.abstract(.f32, &.{ in_dim, h1 }),
        .b1 = Tensor.abstract(.f32, &.{h1}),
        .w2 = Tensor.abstract(.f32, &.{ h1, h2 }),
        .b2 = Tensor.abstract(.f32, &.{h2}),
        .w3 = Tensor.abstract(.f32, &.{ h2, out_dim }),
        .b3 = Tensor.abstract(.f32, &.{out_dim}),
    };
    const batch_spec = BatchSpec{
        .x = Tensor.abstract(.f32, &.{ bs, in_dim }),
        .y = Tensor.abstract(.f32, &.{ bs, out_dim }),
    };
    const inputs_spec = .{ params_spec, batch_spec };
    const donate = comptime zg.frontend.train.donate_argnums(@TypeOf(inputs_spec), &.{0});

    const lower_encoding: zg.pipeline.MlirEncoding = if (dump_mlir != null) .text else .bytecode;

    const train = zg.frontend.train;
    var program = try zg.trace(Fns.train_step, allocator, inputs_spec, "train_step");
    defer program.deinit();

    const exe = try zg.frontend.compile_program(backend, allocator, &program, device, "train_step", .{
        .lower = .{ .encoding = lower_encoding },
        .dump_pr = if (dump_pr) |cfg| cfg.* else null,
        .dump_mlir = if (dump_mlir) |cfg| cfg.* else null,
        .dump_optimized = if (dump_optimized) |cfg| cfg.* else null,
    });
    defer backend.deinit_executable(exe);

    const true_w1 = try allocator.alloc(f32, @intCast(in_dim * h1));
    defer allocator.free(true_w1);
    const true_b1 = try allocator.alloc(f32, @intCast(h1));
    defer allocator.free(true_b1);
    const true_w2 = try allocator.alloc(f32, @intCast(h1 * h2));
    defer allocator.free(true_w2);
    const true_b2 = try allocator.alloc(f32, @intCast(h2));
    defer allocator.free(true_b2);
    const true_w3 = try allocator.alloc(f32, @intCast(h2 * out_dim));
    defer allocator.free(true_w3);
    const true_b3 = try allocator.alloc(f32, @intCast(out_dim));
    defer allocator.free(true_b3);

    fill_pattern(true_w1, 1e-6, 0.0);
    fill_pattern(true_b1, 1e-6, 0.0);
    fill_pattern(true_w2, 1e-6, 0.0);
    fill_pattern(true_b2, 1e-6, 0.0);
    fill_pattern(true_w3, 1e-6, 0.0);
    fill_pattern(true_b3, 1e-6, 0.0);

    // Build host tensors from tree
    var spec_tree = try zg.utils.Tree(Tensor).from(allocator, inputs_spec);
    defer spec_tree.deinit();

    var host_tensors = try spec_tree.map(Tensor, allocator, struct {
        fn f(alloc: std.mem.Allocator, spec: Tensor) !Tensor {
            return try Tensor.host(spec.dtype, spec.shape.const_slice(), .{ .alloc = alloc });
        }
    }.f);
    defer host_tensors.deinit_with(deinit_tensor);

    // Fill param buffers with pattern, batch input with data
    for (host_tensors.leaves[0..6]) |*buf| fill_pattern(buf.as_slice(f32), 1e-7, 0.0);
    fill_inputs(host_tensors.leaves[6].as_slice(f32));

    // Generate targets from true weights
    const scratch1 = try allocator.alloc(f32, @intCast(bs * h1));
    defer allocator.free(scratch1);
    const scratch2 = try allocator.alloc(f32, @intCast(bs * h2));
    defer allocator.free(scratch2);
    try fill_targets(
        host_tensors.leaves[7],
        host_tensors.leaves[6],
        true_w1,
        true_b1,
        true_w2,
        true_b2,
        true_w3,
        true_b3,
        scratch1,
        scratch2,
        @intCast(bs),
        @intCast(in_dim),
        @intCast(h1),
        @intCast(h2),
        @intCast(out_dim),
    );

    // Upload to device
    const UploadCtx = struct { b: *zg.Backend, d: zg.Backend.Device };
    var dev_tree = try host_tensors.map(Tensor, UploadCtx{ .b = backend, .d = device }, struct {
        fn f(ctx: UploadCtx, t: Tensor) !Tensor {
            return try t.to_device(ctx.b, ctx.d);
        }
    }.f);
    defer dev_tree.deinit(); // array only, tensor ownership managed by TrainState

    var state = try train.TrainState.init(
        allocator,
        exe,
        backend,
        dev_tree.leaves,
        program.output_arity("train_step"),
        .{ .non_donatable_input_indices = donate },
    );
    defer state.deinit(.all);

    for (0..warmup_steps) |_| {
        const result = try state.step();
        // Deinit execution event without awaiting. buffer_to_host (called by
        //  item below) chains behind execution internally.
        // TODO: verify PJRT_Event_Destroy on non-awaited event is spec-safe.
        if (result.event) |ev| backend.deinit_event(ev);
        if (!quiet) _ = try result.loss.item(f32);
        result.loss.deinit();
    }

    var loop_timer = zg.utils.LoopTimer{ .label = "train-demo", .quiet = quiet };
    for (0..steps) |_| {
        try loop_timer.start_step();
        const result = try state.step();
        // Deinit execution event without awaiting. item() below syncs via
        //  buffer_to_host which chains behind execution.
        // TODO: verify PJRT_Event_Destroy on non-awaited event is spec-safe.
        if (result.event) |ev| backend.deinit_event(ev);
        loop_timer.mark("dispatch");

        const loss: ?f32 = if (quiet) null else try result.loss.item(f32);
        loop_timer.mark("sync+read");

        result.loss.deinit();
        loop_timer.mark("cleanup");

        loop_timer.end_step(loss);
    }

    log.info("avg_step_ms={d:.3}", .{loop_timer.avg_ms()});
    log.info("OK", .{});
}

/// End-to-end kernel provider demo.
///
/// Compiles and executes a small matmul program through the kernelization
///  pipeline. Accepts one or more providers, each gets its own kernelized
///  region in the program.
///
/// TVM requires libtvm in the SDK lib/ directory, Mirage requires the
///  Mirage shared library. Both can be enabled individually or simultaneously.
/// Run the kernel provider demo: tune -> store -> compile -> execute.
///
/// Always uses the store-based (PR-level) path: `tune()` populates a
/// `KernelStore`, `KernelizePass` rewrites annotated regions, and the
/// backend dispatches via `DispatchRegistry` at execute time.
pub fn run_kernel_provider_demo(
    allocator: std.mem.Allocator,
    pjrt_backend: *zg.backend.pjrt.Backend,
    device: zg.Backend.Device,
    dump_pr: ?*zg.pipeline.DumpConfig,
    dump_mlir: ?*zg.pipeline.DumpConfig,
    dump_optimized: ?*zg.pipeline.DumpConfig,
    provider_kinds: []const KernelProviderDemoKind,
) !void {
    const backend = &pjrt_backend.interface;
    try pjrt_backend.register_kernel_dispatcher();

    const demo_cache = try zg.Cache.init(.{});

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
        try pjrt_backend.require_typed_ffi();
        const target_kind: zg.tvm.tir.TargetKind = if (pjrt_backend.is_cuda()) .cuda else .cpu;
        tvm_dispatch = zg.tvm.dispatch.TvmDispatchState.init(allocator, demo_cache);
        tvm_impl = .{
            .allocator = allocator,
            .target_kind = target_kind,
            .cache = demo_cache,
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

    // tune -> store -> compile
    var tune_result = try zg.tune.tune(allocator, &program, providers, .{});
    defer tune_result.deinit();

    const exe = try zg.frontend.compile_program(backend, allocator, &program, device, "main", .{
        .lower = .{ .encoding = lower_encoding },
        .kernel_store = &tune_result.store,
        .dump_pr = if (dump_pr) |cfg| cfg.* else null,
        .dump_mlir = if (dump_mlir) |cfg| cfg.* else null,
        .dump_optimized = if (dump_optimized) |cfg| cfg.* else null,
    });
    defer backend.deinit_executable(exe);

    const exec_opts: zg.Backend.ExecuteOptions = .{
        .store = &tune_result.store,
        .dispatch_registry = &tune_result.dispatch_registry,
    };

    return run_kernel_provider_demo_executable(allocator, backend, device, exe, provider_kinds.len, exec_opts);
}

fn kind_requested(kinds: []const KernelProviderDemoKind, target: KernelProviderDemoKind) bool {
    for (kinds) |k| if (k == target) return true;
    return false;
}

/// Execute the kernel provider demo program and verify results.
fn run_kernel_provider_demo_executable(
    allocator: std.mem.Allocator,
    backend: *zg.Backend,
    device: zg.Backend.Device,
    exe: zg.Backend.Executable,
    n_providers: usize,
    exec_opts: zg.Backend.ExecuteOptions,
) !void {
    const A = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    const B = [_]f32{ 7.0, 8.0, 9.0, 10.0, 11.0, 12.0 };
    const C = [_]f32{ 2.0, 2.0, 2.0, 2.0 };

    const host_a = try Tensor.host(.f32, &.{ 2, 3 }, .{ .borrow = std.mem.sliceAsBytes(&A) });
    defer host_a.deinit();
    const host_b = try Tensor.host(.f32, &.{ 3, 2 }, .{ .borrow = std.mem.sliceAsBytes(&B) });
    defer host_b.deinit();
    const host_c = try Tensor.host(.f32, &.{ 2, 2 }, .{ .borrow = std.mem.sliceAsBytes(&C) });
    defer host_c.deinit();

    const dev_a = try backend.buffer_from_host(device, host_a.host_data(), .f32, host_a.dims());
    defer backend.deinit_buffer(dev_a);
    const dev_b = try backend.buffer_from_host(device, host_b.host_data(), .f32, host_b.dims());
    defer backend.deinit_buffer(dev_b);
    const dev_c = try backend.buffer_from_host(device, host_c.host_data(), .f32, host_c.dims());
    defer backend.deinit_buffer(dev_c);

    const result = try backend.execute(exe, allocator, &.{ dev_a, dev_b, dev_c }, exec_opts);
    defer {
        if (result.event) |ev| backend.deinit_event(ev);
        for (result.outputs) |buf| backend.deinit_buffer(buf);
        allocator.free(result.outputs);
    }

    if (result.outputs.len != 1) return error.UnexpectedOutputs;

    var out_host = try Tensor.host(.f32, &.{ 2, 2 }, .{ .alloc = allocator });
    defer out_host.deinit();
    if (try backend.buffer_to_host(result.outputs[0], out_host.host_data_mut())) |ev| {
        defer backend.deinit_event(ev);
        try backend.await_event(ev);
    }

    // dot(A, B) = [[58, 64], [139, 154]]
    // (n*dot + C) * C = [[116n+4, 128n+4], [278n+4, 308n+4]]
    const n = @as(f32, @floatFromInt(n_providers));
    const expected = [_]f32{
        116.0 * n + 4.0, 128.0 * n + 4.0,
        278.0 * n + 4.0, 308.0 * n + 4.0,
    };
    const out = out_host.as_const_slice(f32);
    std.debug.assert(out.len == 4);
    for (out, 0..) |v, i| {
        const diff = @abs(v - expected[i]);
        if (diff > 1e-4) {
            std.log.err("mismatch[{d}]: got {d}, expected {d}", .{ i, v, expected[i] });
            return error.NumericalMismatch;
        }
    }
    log.info("OK: output matches expected ({d} provider(s))", .{n_providers});
}

pub const KernelProviderDemoKind = enum {
    tvm,
    mirage,
};

pub fn print_pr(allocator: std.mem.Allocator) !void {
    var program = try build_demo_program(allocator);
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

    try tvm_api.ensure_loaded(allocator, .{ .load_compiler = true });

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

    var builder = try zg.pr.FunctionBuilder.init(&program, "main");
    defer builder.deinit();

    const a = try Tensor.param(&builder, .f32, &.{ 2, 3 });
    const b_t = try Tensor.param(&builder, .f32, &.{ 3, 2 });
    const c = try Tensor.param(&builder, .f32, &.{ 2, 2 });
    const d = try Tensor.param(&builder, .f32, &.{ 2, 2 });

    const pre = try c.add(d);

    try builder.push_region("tvm_matmul", .{ .kernelize = "tvm" });
    const dot = try a.matmul(b_t);
    try builder.pop_region();

    try builder.push_region("tvm_fused", .{ .kernelize = "tvm", .outline = true });
    const sum = try dot.add(pre);
    const mul = try sum.mul(c);
    try builder.pop_region();

    const out = try mul.add(d);
    const out_var = try out.get_var();
    const func_result = try builder.finish(&.{out_var});
    try program.add_function(func_result);

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
    const first_name = try std.fmt.allocPrint(allocator, "{s}_region_0", .{provider_names[0]});
    try b.push_region(first_name, .{ .kernelize = provider_names[0] });
    var acc_id = try b.dot(a_id, b_id);
    try b.pop_region();

    for (provider_names[1..], 1..) |pname, i| {
        const rn = try std.fmt.allocPrint(allocator, "{s}_region_{d}", .{ pname, i });
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
    host_y: Tensor,
    host_x: Tensor,
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
    if (host_x.shape.len != 2 or host_y.shape.len != 2) return error.ShapeMismatch;
    const x_dims = host_x.shape.const_slice();
    const y_dims = host_y.shape.const_slice();
    if (x_dims[0] != @as(i64, @intCast(bs)) or x_dims[1] != @as(i64, @intCast(in_dim))) return error.ShapeMismatch;
    if (y_dims[0] != @as(i64, @intCast(bs)) or y_dims[1] != @as(i64, @intCast(out_dim))) return error.ShapeMismatch;

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

    var builder = try zg.pr.FunctionBuilder.init(&program, "attention");
    defer builder.deinit();

    // simplified attention: [B, S, D] shapes
    // Q, K, V: [batch=2, seq=4, head_dim=64]
    const batch: i64 = 2;
    const seq: i64 = 4;
    const head_dim: i64 = 64;

    const q = try Tensor.param(&builder, .f32, &.{ batch, seq, head_dim });
    const k = try Tensor.param(&builder, .f32, &.{ batch, seq, head_dim });
    const v = try Tensor.param(&builder, .f32, &.{ batch, seq, head_dim });

    // entire attention block as a single TVM-kernelizable region
    try builder.push_region("attention", .{ .kernelize = "tvm" });

    // attention scores: Q @ K^T  ->  [B, S, S]
    const scores = try q.dot_general(k, .{
        .lhs_batch_dims = &.{0},
        .rhs_batch_dims = &.{0},
        .lhs_contracting_dims = &.{2},
        .rhs_contracting_dims = &.{2},
    });

    // scale scores
    const scale_val = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));
    const scaled = try scores.mul(try Tensor.constant_like(scores, scale_val));

    // softmax over last dim [S]
    const rank = scaled.rank();
    const axis: i64 = @intCast(rank - 1);
    const max_val = try scaled.reduce_max(&.{axis});

    // broadcast max back to full shape for stability
    const max_broadcast = try max_val.broadcast_in_dim(scaled.dims(), &.{ 0, 1 });
    const shifted = try scaled.sub(max_broadcast);
    const exp_vals = try shifted.exp();
    const sum_exp = try exp_vals.reduce_sum(&.{axis});
    const sum_broadcast = try sum_exp.broadcast_in_dim(exp_vals.dims(), &.{ 0, 1 });
    const attn_weights = try exp_vals.div(sum_broadcast);

    // output: attn_weights @ V  ->  [B, S, D]
    const out = try attn_weights.dot_general(v, .{
        .lhs_batch_dims = &.{0},
        .rhs_batch_dims = &.{0},
        .lhs_contracting_dims = &.{2},
        .rhs_contracting_dims = &.{1},
    });

    try builder.pop_region();

    const out_var = try out.get_var();
    const func_result = try builder.finish(&.{out_var});
    try program.add_function(func_result);

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
