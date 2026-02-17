const std = @import("std");
const zg = @import("zigrad");
const stz = @import("safetensors_zg");

pub fn run_llm_ft_demo(
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
        w_emb: TensorSpec,
        w_out: TensorSpec,
        b: TensorSpec,
    };

    const BatchSpec = struct {
        x: TensorSpec,
        y: TensorSpec,
    };

    const LossFn = struct {
        fn call(params: anytype, batch: anytype) !zg.frontend.Tensor {
            const bs: usize = 16;
            const vocab: usize = 128;

            const hidden_act = try batch.x.matmul(params.w_emb);
            const logits = try hidden_act.matmul(params.w_out);
            const bcast_b = try params.b.broadcast_in_dim(&.{ bs, vocab }, &.{1});
            const logits_b = try logits.add(bcast_b);

            const exp_logits = try logits_b.exp();
            const sum_exp = try exp_logits.reduce_sum(&.{1});
            const log_sum = try sum_exp.log();
            const log_sum_b = try log_sum.broadcast_in_dim(&.{ bs, vocab }, &.{0});
            const log_softmax = try logits_b.sub(log_sum_b);

            const y_log = try batch.y.mul(log_softmax);
            const loss_per = try y_log.reduce_sum(&.{1});

            const neg = try log_softmax.builder.scalar_literal(.{ .f32 = -1.0 });
            const neg_b = try neg.broadcast_in_dim(&.{bs}, &.{});
            const neg_loss = try loss_per.mul(neg_b);
            return try neg_loss.reduce_sum(&.{0});
        }
    };

    const bs: usize = 16;
    const vocab: usize = 128;
    const hidden: usize = 64;

    const params_spec = ParamsSpec{
        .w_emb = .{ .dtype = .f32, .dims = &.{ vocab, hidden } },
        .w_out = .{ .dtype = .f32, .dims = &.{ hidden, vocab } },
        .b = .{ .dtype = .f32, .dims = &.{vocab} },
    };
    const batch_spec = BatchSpec{
        .x = .{ .dtype = .f32, .dims = &.{ bs, vocab } },
        .y = .{ .dtype = .f32, .dims = &.{ bs, vocab } },
    };
    const inputs_spec = .{ params_spec, batch_spec };

    var compile_cfg = zg.frontend.CompileConfig{
        .entry_name = "llm_ft_step",
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
    var compiled = try train.compile_train_step(allocator, &backend_handle, device, LossFn.call, inputs_spec, 3, .{
        .optimizer = .{ .lr = 1e-2 },
        .compile = compile_cfg,
    });
    defer compiled.exe.deinit(backend_handle.api);

    const shape_w_emb = zg.utils.Shape{ .dims = &.{ vocab, hidden } };
    const shape_w_out = zg.utils.Shape{ .dims = &.{ hidden, vocab } };
    const shape_b = zg.utils.Shape{ .dims = &.{vocab} };
    const shape_x = zg.utils.Shape{ .dims = &.{ bs, vocab } };
    const shape_y = zg.utils.Shape{ .dims = &.{ bs, vocab } };

    var host_w_emb = try zg.utils.HostBuffer.init(allocator, shape_w_emb, .f32);
    defer host_w_emb.deinit();
    var host_w_out = try zg.utils.HostBuffer.init(allocator, shape_w_out, .f32);
    defer host_w_out.deinit();
    var host_b = try zg.utils.HostBuffer.init(allocator, shape_b, .f32);
    defer host_b.deinit();
    var host_x = try zg.utils.HostBuffer.init(allocator, shape_x, .f32);
    defer host_x.deinit();
    var host_y = try zg.utils.HostBuffer.init(allocator, shape_y, .f32);
    defer host_y.deinit();

    if (std.process.getEnvVarOwned(allocator, "ZG_LLM_SAFETENSORS_PATH")) |path| {
        defer allocator.free(path);
        try load_safetensors_weights(
            allocator,
            path,
            host_w_emb.as_slice(f32),
            host_w_out.as_slice(f32),
            host_b.as_slice(f32),
            shape_w_emb.dims,
            shape_w_out.dims,
            shape_b.dims,
        );
        if (!quiet) {
            std.log.info("llm-ft-demo: loaded weights from {s}", .{path});
        }
    } else |_| {
        fill_pattern(host_w_emb.as_slice(f32), 1e-3, 0.0);
        fill_pattern(host_w_out.as_slice(f32), 1e-3, 0.0);
        fill_pattern(host_b.as_slice(f32), 1e-3, 0.0);
    }

    const tokens = try allocator.alloc(usize, bs);
    defer allocator.free(tokens);
    const targets = try allocator.alloc(usize, bs);
    defer allocator.free(targets);
    fill_token_pairs(tokens, targets, vocab);
    fill_one_hot(host_x.as_slice(f32), tokens, vocab);
    fill_one_hot(host_y.as_slice(f32), targets, vocab);

    var total_ns: u64 = 0;

    const upload = zg.frontend.upload_host_buffer;
    const tmp_w_emb = try upload(allocator, &backend_handle, device, &host_w_emb);
    const tmp_w_out = try upload(allocator, &backend_handle, device, &host_w_out);
    const tmp_b = try upload(allocator, &backend_handle, device, &host_b);
    const tmp_x = try upload(allocator, &backend_handle, device, &host_x);
    const tmp_y = try upload(allocator, &backend_handle, device, &host_y);

    var loss_host = try zg.utils.HostBuffer.init(allocator, .{ .dims = &.{} }, .f32);
    defer loss_host.deinit();

    const api = backend_handle.api;

    var state = try train.TrainState.init(
        allocator,
        &compiled,
        api,
        &.{ tmp_w_emb.pjrt_buffer, tmp_w_out.pjrt_buffer, tmp_b.pjrt_buffer },
        &.{ tmp_x.pjrt_buffer, tmp_y.pjrt_buffer },
    );
    defer state.deinit();
    defer {
        var bx = zg.backend.pjrt.Buffer{ .pjrt_buffer = tmp_x.pjrt_buffer };
        bx.deinit(api);
        var by = zg.backend.pjrt.Buffer{ .pjrt_buffer = tmp_y.pjrt_buffer };
        by.deinit(api);
    }

    const is_cpu = try (zg.backend.pjrt.Buffer{ .pjrt_buffer = tmp_w_emb.pjrt_buffer }).is_on_cpu(api);

    var warmup: usize = 0;
    while (warmup < warmup_steps) : (warmup += 1) {
        var result = try state.step();
        if (result.event) |e| {
            var ev = e;
            try ev.await_(api);
            ev.deinit(api);
        }
        if (!is_cpu and !quiet) {
            var loss_ev = try result.loss_buf.to_host(api, loss_host.data);
            try loss_ev.await_(api);
            loss_ev.deinit(api);
        }
        result.loss_buf.deinit(api);
    }

    var step: usize = 0;
    while (step < steps) : (step += 1) {
        var timer = try std.time.Timer.start();
        var result = try state.step();
        const dispatch_ns = timer.lap();

        if (result.event) |e| {
            var ev = e;
            try ev.await_(api);
            ev.deinit(api);
        }
        const wait_ns = timer.lap();

        const loss: ?f32 = if (quiet) null else if (is_cpu) blk: {
            const ptr: [*]const f32 = @ptrFromInt(try result.loss_buf.unsafe_pointer(api));
            break :blk ptr[0];
        } else blk: {
            var loss_ev = try result.loss_buf.to_host(api, loss_host.data);
            try loss_ev.await_(api);
            loss_ev.deinit(api);
            break :blk loss_host.as_slice(f32)[0];
        };
        const loss_read_ns = timer.lap();

        result.loss_buf.deinit(api);

        const cleanup_ns = timer.lap();
        const step_ns = dispatch_ns + wait_ns + loss_read_ns + cleanup_ns;
        total_ns += step_ns;
        const step_ms = @as(f64, @floatFromInt(step_ns)) / std.time.ns_per_ms;
        const dispatch_ms = @as(f64, @floatFromInt(dispatch_ns)) / std.time.ns_per_ms;
        const wait_ms = @as(f64, @floatFromInt(wait_ns)) / std.time.ns_per_ms;
        const loss_ms = @as(f64, @floatFromInt(loss_read_ns)) / std.time.ns_per_ms;
        const cleanup_ms = @as(f64, @floatFromInt(cleanup_ns)) / std.time.ns_per_ms;
        if (!quiet) {
            std.log.info("llm-ft-demo step {d}: loss={d:.6} dispatch={d:.3}ms wait={d:.3}ms loss={d:.3}ms cleanup={d:.3}ms total={d:.3}ms", .{
                step, loss.?, dispatch_ms, wait_ms, loss_ms, cleanup_ms, step_ms,
            });
        }
    }

    const avg_ms = @as(f64, @floatFromInt(total_ns)) / std.time.ns_per_ms / @as(f64, @floatFromInt(steps));
    std.log.info("llm-ft-demo avg_step_ms={d:.3}", .{avg_ms});
    std.log.info("OK: llm-ft-demo executed", .{});
}

fn fill_pattern(slice: []f32, scale: f32, offset: f32) void {
    for (slice, 0..) |*v, i| {
        const base = @as(f32, @floatFromInt(i % 1024));
        v.* = offset + scale * base;
    }
}

fn fill_token_pairs(tokens: []usize, targets: []usize, vocab: usize) void {
    var state: u64 = 0x9e3779b97f4a7c15;
    for (tokens, 0..) |*t, i| {
        state = state *% 6364136223846793005 +% 1;
        t.* = @intCast(state % vocab);
        state = state *% 6364136223846793005 +% 1;
        targets[i] = @intCast(state % vocab);
    }
}

fn fill_one_hot(out: []f32, indices: []const usize, vocab: usize) void {
    @memset(out, 0);
    for (indices, 0..) |idx, i| {
        const offset = i * vocab + idx;
        if (offset < out.len) out[offset] = 1.0;
    }
}

fn load_safetensors_weights(
    allocator: std.mem.Allocator,
    path: []const u8,
    w_emb: []f32,
    w_out: []f32,
    b: []f32,
    shape_w_emb: []const usize,
    shape_w_out: []const usize,
    shape_b: []const usize,
) !void {
    const data = try read_file_aligned(allocator, path);
    defer allocator.free(data);

    var st_file = try stz.SafeTensorsFile.deserialize(data, allocator);
    defer st_file.deinit();

    const w_emb_view = try st_file.get("w_emb");
    const w_out_view = try st_file.get("w_out");
    const b_view = try st_file.get("b");

    try copy_tensor_f32(w_emb_view, w_emb, shape_w_emb);
    try copy_tensor_f32(w_out_view, w_out, shape_w_out);
    try copy_tensor_f32(b_view, b, shape_b);
}

fn read_file_aligned(allocator: std.mem.Allocator, path: []const u8) ![]align(8) u8 {
    var file = if (std.fs.path.isAbsolute(path))
        try std.fs.openFileAbsolute(path, .{})
    else
        try std.fs.cwd().openFile(path, .{});
    defer file.close();

    const stat = try file.stat();
    const size: usize = @intCast(stat.size);
    const buf = try allocator.alignedAlloc(u8, .@"8", size);
    const read_len = try file.readAll(buf);
    if (read_len != size) return error.UnexpectedEof;
    return buf;
}

fn copy_tensor_f32(view: stz.TensorView, out: []f32, expected_shape: []const usize) !void {
    if (view.info.dtype != .f32) return error.TensorDtypeMismatch;
    if (!std.mem.eql(usize, view.info.shape, expected_shape)) return error.TensorShapeMismatch;

    var count: usize = 1;
    for (view.info.shape) |d| count *= d;
    if (count != out.len) return error.TensorSizeMismatch;

    const data = std.mem.bytesAsSlice(f32, view.data);
    if (data.len != out.len) return error.TensorSizeMismatch;
    @memcpy(out, data);
}
