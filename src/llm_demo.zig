const std = @import("std");
const zg = @import("zigrad");
const stz = @import("safetensors_zg");
const log = std.log.scoped(.@"zg/llm_demo");

pub fn run_llm_train_demo(
    ctx: *zg.CompilationCtx,
    pipeline: *zg.Pipeline,
    environ: *const std.process.Environ.Map,
    warmup_steps: usize,
    steps: usize,
    quiet: bool,
) !void {
    const io = ctx.io;
    const allocator = ctx.allocator;
    const Tensor = zg.Tensor;

    const ParamsSpec = struct {
        w_emb: Tensor,
        w_out: Tensor,
        b: Tensor,
    };

    const BatchSpec = struct {
        x: Tensor,
        y: Tensor,
    };

    const Fns = struct {
        fn loss(params: ParamsSpec, batch: BatchSpec) !Tensor {
            const bs_: i64 = 16;
            const vocab_: i64 = 128;

            const hidden_act = try batch.x.mm(params.w_emb);
            const logits = try hidden_act.mm(params.w_out);
            const bcast_b = try params.b.broadcast_in_dim(&.{ bs_, vocab_ }, &.{1});
            const logits_b = try logits.add(bcast_b);

            const exp_logits = try logits_b.exp();
            const sum_exp = try exp_logits.reduce(.{ .axes = &.{1}, .operation = .sum });
            const log_sum = try sum_exp.log();
            const log_sum_b = try log_sum.broadcast_in_dim(&.{ bs_, vocab_ }, &.{0});
            const log_softmax = try logits_b.sub(log_sum_b);

            const y_log = try batch.y.mul(log_softmax);
            const loss_per = try y_log.reduce(.{ .axes = &.{1}, .operation = .sum });

            const neg_loss = try loss_per.mul(try Tensor.constant_like(loss_per, -1.0));
            return try neg_loss.reduce(.{ .axes = &.{0}, .operation = .sum });
        }

        fn train_step(params: ParamsSpec, batch: BatchSpec) !struct { loss_val: Tensor, updated: ParamsSpec } {
            var vg = try zg.transforms.value_and_grad(loss, .{ params, batch });
            defer vg.deinit();
            var params_tree = try zg.utils.Tree(Tensor).from(vg.grads.allocator, params);
            defer params_tree.deinit();
            const optim = zg.optim.SGD{ .lr = 1e-2 };
            var updated = try params_tree.map2(Tensor, &vg.grads, Tensor, optim, zg.optim.SGD.update);
            defer updated.deinit();
            return .{
                .loss_val = vg.value,
                .updated = try updated.extract(ParamsSpec),
            };
        }
    };

    const bs: i64 = 16;
    const vocab: i64 = 128;
    const hidden: i64 = 64;

    const params_spec = ParamsSpec{
        .w_emb = Tensor.abstract(.f32, &.{ vocab, hidden }),
        .w_out = Tensor.abstract(.f32, &.{ hidden, vocab }),
        .b = Tensor.abstract(.f32, &.{vocab}),
    };
    const batch_spec = BatchSpec{
        .x = Tensor.abstract(.f32, &.{ bs, vocab }),
        .y = Tensor.abstract(.f32, &.{ bs, vocab }),
    };
    const inputs_spec = .{ params_spec, batch_spec };
    const donate = comptime zg.train.donate_argnums(@TypeOf(inputs_spec), &.{0});

    const train = zg.train;
    var program = try zg.trace(Fns.train_step, allocator, inputs_spec, "llm_ft_step");
    defer program.deinit();

    var exe = try pipeline.run(
        zg.Executor.LoadedProgram,
        &program,
        ctx,
    );
    defer exe.deinit();
    const executor = exe.executor;

    var host_w_emb = try Tensor.host(.f32, &.{ vocab, hidden }, .{ .alloc = allocator });
    defer host_w_emb.deinit();
    var host_w_out = try Tensor.host(.f32, &.{ hidden, vocab }, .{ .alloc = allocator });
    defer host_w_out.deinit();
    var host_b = try Tensor.host(.f32, &.{vocab}, .{ .alloc = allocator });
    defer host_b.deinit();
    var host_x = try Tensor.host(.f32, &.{ bs, vocab }, .{ .alloc = allocator });
    defer host_x.deinit();
    var host_y = try Tensor.host(.f32, &.{ bs, vocab }, .{ .alloc = allocator });
    defer host_y.deinit();

    if (environ.get("ZG_LLM_SAFETENSORS_PATH")) |path| {
        try load_safetensors_weights(
            allocator,
            path,
            host_w_emb.as_slice(f32),
            host_w_out.as_slice(f32),
            host_b.as_slice(f32),
            host_w_emb.shape.const_slice(),
            host_w_out.shape.const_slice(),
            host_b.shape.const_slice(),
        );
        if (!quiet) {
            log.info("Loaded weights from {s}", .{path});
        }
    } else {
        log.warn("Using synthetic weights (set ZG_LLM_SAFETENSORS_PATH to use a specific checkpoint)", .{});
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

    const dev_w_emb = try host_w_emb.to_device(executor);
    const dev_w_out = try host_w_out.to_device(executor);
    const dev_b = try host_b.to_device(executor);
    const dev_x = try host_x.to_device(executor);
    const dev_y = try host_y.to_device(executor);

    var state = try train.TrainState.init(
        allocator,
        exe,
        &.{ dev_w_emb, dev_w_out, dev_b, dev_x, dev_y },
        program.output_arity("llm_ft_step"),
        .{ .non_donatable_input_indices = donate },
    );
    defer state.deinit(.all);

    for (0..warmup_steps) |_| {
        var result = try state.step();
        // TODO(pjrt): Verify that releasing an unawaited event is valid.
        if (result.event) |completion| executor.release_event(completion);
        result.loss.deinit();
    }

    var loop_timer = zg.utils.LoopTimer{ .io = io, .label = "llm-train", .quiet = quiet };
    for (0..steps) |_| {
        try loop_timer.start_step();
        var result = try state.step();
        // TODO(pjrt): Verify that releasing an unawaited event is valid.
        if (result.event) |completion| executor.release_event(completion);
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
    shape_w_emb: []const i64,
    shape_w_out: []const i64,
    shape_b: []const i64,
) !void {
    const data = try zg.utils.mmap_file(path);
    defer zg.utils.munmap(data);

    var st_file = try stz.SafeTensorsFile.deserialize(data, allocator);
    defer st_file.deinit();

    const w_emb_view = try st_file.get("w_emb");
    const w_out_view = try st_file.get("w_out");
    const b_view = try st_file.get("b");

    try copy_tensor_f32(w_emb_view, w_emb, shape_w_emb);
    try copy_tensor_f32(w_out_view, w_out, shape_w_out);
    try copy_tensor_f32(b_view, b, shape_b);
}

fn copy_tensor_f32(view: stz.TensorView, out: []f32, expected_shape: []const i64) !void {
    if (view.info.dtype != .f32) return error.TensorDtypeMismatch;
    // Compare safetensors usize dimensions with PR i64 dimensions.
    if (view.info.shape.len != expected_shape.len) return error.TensorShapeMismatch;
    for (view.info.shape, expected_shape) |a, b| {
        if (a != @as(usize, @intCast(b))) return error.TensorShapeMismatch;
    }

    var count: usize = 1;
    for (view.info.shape) |d| count *= d;
    if (count != out.len) return error.TensorSizeMismatch;

    const data = std.mem.bytesAsSlice(f32, view.data);
    if (data.len != out.len) return error.TensorSizeMismatch;
    @memcpy(out, data);
}
