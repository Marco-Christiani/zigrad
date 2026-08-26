//! LLaMA checkpoint loading, tracing, compilation, and execution.

const std = @import("std");
const zg = @import("zigrad");

const model = @import("model.zig");
const Tensor = zg.Tensor;
const Backend = zg.Backend(zg.stablehlo.Artifact);

const log = std.log.scoped(.@"zg/llama_training");

/// Create an iota tensor from a traced tensor's builder.
///
/// TODO(api): Expose this operation through `Tensor`.
fn iota_from(t: Tensor, out_dtype: zg.DType, out_dims: []const i64, iota_dim: i64) !Tensor {
    const b = t.backing.traced.builder;
    return Tensor.from_var(b, try b.iota(out_dtype, out_dims, iota_dim));
}

const num_layers: usize = 16;

/// Per-attention-head dimension.
///
/// This value matches the LLaMA 3.2 1B checkpoint used by the example.
///
/// TODO(example): Read model dimensions from the checkpoint configuration.
const head_dim: i64 = 64;

/// Checkpoint-shaped model parameters.
const LlamaParams = model.LlamaWeights(num_layers);

const Batch = struct {
    x: Tensor,
    target_ids: Tensor,
    attention_mask: Tensor,
    mask: Tensor,
    sin: Tensor,
    cos: Tensor,
};

/// Program compiled and executed by the example.
pub const Mode = enum {
    training,
    inference,
};

/// Inputs controlling checkpoint loading and execution.
pub const Options = struct {
    /// Select the compiled program.
    mode: Mode = .training,
    /// Element type used for model parameters and activations.
    dtype: zg.DType = .bf16,
    /// Number of tokens in each batch row.
    sequence_length: i64 = 4,
    /// Number of batch rows.
    batch: i64 = 1,
    /// Number of untimed executions before measurement.
    warmup_steps: usize = 5,
    /// Number of measured executions.
    steps: usize = 20,
    /// Suppress per-step timing output.
    quiet: bool = false,
    /// SafeTensors checkpoint whose keys match `LlamaParams`.
    weights_path: []const u8,
};

const TrainStepResult = struct { loss_val: Tensor, updated: LlamaParams };

fn train_step_fn(params: LlamaParams, batch: Batch) !TrainStepResult {
    var vg = try zg.transforms.value_and_grad(loss_fn, .{ params, batch }, .{});
    defer vg.deinit();
    return try sgd_step(params, &vg, 1e-4);
}

fn sgd_step(params: LlamaParams, vg: *zg.transforms.ValueAndGrad(Tensor), lr: f32) !TrainStepResult {
    var params_tree = try zg.utils.Tree(Tensor).from(vg.grads.allocator, params);
    defer params_tree.deinit();
    const optim = zg.optim.SGD{ .lr = lr };
    var updated = try params_tree.map2(Tensor, &vg.grads, Tensor, optim, zg.optim.SGD.update);
    defer updated.deinit();
    return .{
        .loss_val = vg.outputs,
        .updated = try updated.extract(LlamaParams),
    };
}

fn loss_fn(params: LlamaParams, batch: Batch) !Tensor {
    const batch_size: i64 = batch.x.dims()[0];
    const seq: i64 = batch.x.dims()[1];
    const logits = try params.model.forward(
        batch.x,
        batch.mask,
        batch.attention_mask,
        batch.sin,
        batch.cos,
        1e-6,
    );
    // accumulate loss in f32 bc bf16 logits over the full vocab can overflow in logsumexp
    const logits_f = try logits.convert(.f32);
    const attn_mask = try batch.attention_mask.convert(logits_f.dtype);

    // gather one vocab entry per (batch, seq) row without flattening
    const row_b = try iota_from(logits_f, .i32, &.{ batch_size, seq }, 0);
    const row_s = try iota_from(logits_f, .i32, &.{ batch_size, seq }, 1);
    const tgt_ids = try batch.target_ids.convert(.i32);
    const row_b3 = try row_b.reshape(&.{ batch_size, seq, 1 });
    const row_s3 = try row_s.reshape(&.{ batch_size, seq, 1 });
    const tgt_ids3 = try tgt_ids.reshape(&.{ batch_size, seq, 1 });
    const gather_idx = try row_b3.concatenate(&.{ row_s3, tgt_ids3 }, 2);
    const gather_params: zg.pr.GatherParams = .{
        .slice_sizes = &.{ 1, 1, 1 },
        .offset_dims = &.{},
        .collapsed_slice_dims = &.{ 0, 1, 2 },
        .start_index_map = &.{ 0, 1, 2 },
        .index_vector_dim = 2,
    };
    const target_logits_2d = try logits_f.gather(gather_idx, gather_params);

    const max_logits = try logits_f.reduce(.{ .axes = &.{2}, .operation = .maximum });
    const max_b = try max_logits.broadcast_in_dim(logits_f.dims(), &.{ 0, 1 });
    const shifted = try logits_f.sub(max_b);

    // jax-style logsumexp lowering: shift in f32, exp in the model dtype (bf16 round-trip
    //  preserves jax numerics), accumulate in f32
    const shifted_f32 = try shifted.convert(.f32);
    const exp_logits = try (try shifted_f32.convert(logits.dtype)).exp();
    const exp_logits_f32 = try exp_logits.convert(.f32);
    const sum_exp = try exp_logits_f32.reduce(.{ .axes = &.{2}, .operation = .sum });
    const log_sum = try sum_exp.log();
    const logsumexp = try log_sum.add(try max_logits.convert(log_sum.dtype));

    const loss_per = try logsumexp.sub(try target_logits_2d.convert(logsumexp.dtype));
    const loss_per_out = try loss_per.convert(logits_f.dtype);

    const zero_b = try Tensor.constant_like(attn_mask, 0.0);
    const attn_zero = try attn_mask.compare(zero_b, .{ .direction = .GT, .compare_type = .FLOAT });
    const masked = try loss_per_out.select(attn_zero, zero_b);
    const masked_f = try masked.convert(.f32);
    const loss_sum = try masked_f.reduce(.{ .axes = &.{ 0, 1 }, .operation = .sum });

    // Normalize by token count: loss / max(mask.sum(), 1)
    const attn_zero_f = try attn_zero.convert(.f32);
    const mask_count = try attn_zero_f.reduce(.{ .axes = &.{ 0, 1 }, .operation = .sum });
    const one = try Tensor.constant_like(mask_count, 1.0);
    const denom = try mask_count.max(one);
    const loss_norm = try loss_sum.div(denom);

    return loss_norm;
}

/// Load a checkpoint, compile the selected program, and execute it.
pub fn run(
    /// Compilation services and selected device.
    ctx: *zg.CompilationCtx,
    /// Terminal backend used after the shared PR to StableHLO lowering.
    backend: *Backend,
    /// Checkpoint, shape, and iteration configuration.
    options: Options,
) !void {
    const io = ctx.io;
    const allocator = ctx.allocator;
    if (options.batch <= 0 or options.sequence_length <= 0)
        return error.InvalidShape;

    var pipeline = zg.Pipeline.init(allocator);
    defer pipeline.deinit();
    try zg.mlir.stablehlo.pipeline.add(&pipeline, .{});
    try pipeline.add(backend);

    const training_mode = options.mode == .training;
    const model_dtype = options.dtype;
    const batch_size = options.batch;
    const seq = options.sequence_length;
    log.info("mode={t} dtype={t} batch={d} sequence_length={d}", .{
        options.mode,
        model_dtype,
        batch_size,
        seq,
    });

    const mmap_data = zg.utils.mmap_file(options.weights_path) catch |err| {
        log.err("failed to mmap checkpoint at '{s}': {s}", .{ options.weights_path, @errorName(err) });
        return err;
    };
    defer zg.utils.munmap(mmap_data);

    var st_file = try zg.SafetensorsFile.deserialize(mmap_data, allocator);
    defer st_file.deinit();

    const host_params = try zg.from_safetensors(LlamaParams, &st_file, .{
        .allocator = allocator,
        .dtype = model_dtype,
    });
    if (!options.quiet) log.info("loaded weights from {s}", .{options.weights_path});

    // batch host tensors, allocated directly from the known shapes
    // sin/cos tables hold the rotary frequencies for `head_dim/2` channels per
    //  position, so their second dim is derived from `head_dim`
    const rope_half = @divExact(head_dim, 2);
    const dims_seq_seq: [2]i64 = .{ seq, seq };
    const dims_seq_half: [2]i64 = .{ seq, rope_half };
    const dims_b_s: [2]i64 = .{ batch_size, seq };
    const host_batch = Batch{
        .x = try Tensor.host(.i32, &dims_b_s, .{ .alloc = allocator }),
        .target_ids = try Tensor.host(.i32, &dims_b_s, .{ .alloc = allocator }),
        .attention_mask = try Tensor.host(model_dtype, &dims_b_s, .{ .alloc = allocator }),
        .mask = try Tensor.host(model_dtype, &dims_seq_seq, .{ .alloc = allocator }),
        .sin = try Tensor.host(model_dtype, &dims_seq_half, .{ .alloc = allocator }),
        .cos = try Tensor.host(model_dtype, &dims_seq_half, .{ .alloc = allocator }),
    };

    const inputs = .{ host_params, host_batch };

    var program = zg.pr.Program.init(allocator);
    defer program.deinit();
    const entry = if (training_mode)
        try zg.trace_into(train_step_fn, allocator, &program, inputs, "main")
    else
        try zg.trace_into(loss_fn, allocator, &program, inputs, "main");
    try program.set_entry(entry);
    var loaded_program = try pipeline.run(zg.Executor.LoadedProgram, &program, ctx);
    defer loaded_program.deinit();
    const executor = loaded_program.executor;

    const donated = comptime zg.train.donated_input_indices(@TypeOf(inputs), &.{0});

    // Fill batch leaves with synthetic inputs.
    const token_seed = [_]usize{ 128000, 128009, 128001, 128008 };
    const target_seed = [_]usize{ 128009, 128001, 128008, 128001 };
    var tokens: [token_seed.len]usize = undefined;
    var targets: [target_seed.len]usize = undefined;
    // Vocab from the loaded embedding shape (torch layout `[vocab, hidden]`).
    const vocab_usize: usize = @intCast(host_params.model.embed_tokens.weight.dims()[0]);
    for (token_seed, 0..) |value, idx| tokens[idx] = value % vocab_usize;
    for (target_seed, 0..) |value, idx| targets[idx] = value % vocab_usize;
    const batch_usize: usize = @intCast(batch_size);
    const seq_usize: usize = @intCast(seq);

    fill_i32_tokens(host_batch.x.as_slice(i32), batch_usize, seq_usize, &tokens);
    fill_i32_tokens(host_batch.target_ids.as_slice(i32), batch_usize, seq_usize, &targets);
    fill_attention_mask(host_batch.attention_mask, batch_usize, seq_usize, tokens.len);
    fill_causal_mask(host_batch.mask, seq_usize);
    fill_rope_tables(host_batch.sin, host_batch.cos, seq_usize, @intCast(head_dim));

    // Combined (params, batch) host tree. Built by flattening the two concrete
    //  structs into leaf copies. `host_tree` becomes the sole owner of every
    //  host tensor: its `deinit_with(Tensor.deinit)` frees all host-backing
    //  allocations, and we must not also deinit `host_params`/`host_batch`
    //  directly (value copies share backing storage).
    var host_tree = try zg.utils.Tree(Tensor).from(allocator, .{ host_params, host_batch });
    defer host_tree.deinit_with(Tensor.deinit);

    // transfer inputs to the selected device
    var dev_tree = try host_tree.map(Tensor, executor, struct {
        fn f(selected: *zg.Executor, tensor: Tensor) (zg.Executor.Error || error{UnsupportedAval})!Tensor {
            return try tensor.to_device(selected);
        }
    }.f);

    var loop_timer = zg.utils.LoopTimer{
        .io = io,
        .label = "llama-training",
        .quiet = options.quiet,
    };

    // Each branch releases `dev_tree` according to whether `TrainState` owns
    //  its tensor leaves.
    if (training_mode) {
        // TrainState.deinit releases the leaf buffers after donation swaps,
        //  this scope releases only the tree arrays and paths
        defer dev_tree.deinit();

        const function = program.get_function_by_id(entry) orelse return error.NoEntry;
        var state = try zg.train.TrainState.init(
            allocator,
            loaded_program,
            dev_tree.leaves,
            function.returns.len,
            .{ .donated_input_indices = donated, .loss_dtype = .f32 },
        );
        defer state.deinit(.all);

        for (0..options.warmup_steps) |_| {
            var result = try state.step();
            errdefer result.loss.deinit();
            defer if (result.event) |completion| executor.release_event(completion);
            _ = try result.loss.item(f32);
            result.loss.deinit();
        }

        for (0..options.steps) |_| {
            try loop_timer.start_step();
            var result = try state.step();
            errdefer result.loss.deinit();
            defer if (result.event) |completion| executor.release_event(completion);
            loop_timer.mark("dispatch");

            const loss = try result.loss.item(f32);
            loop_timer.mark("sync+read");

            result.loss.deinit();
            loop_timer.mark("cleanup");

            loop_timer.end_step(if (options.quiet) null else loss);
        }
    } else {
        defer dev_tree.deinit_with(Tensor.deinit);

        const input_buffers = try allocator.alloc(zg.Executor.Buffer, dev_tree.leaves.len);
        defer allocator.free(input_buffers);
        for (input_buffers, dev_tree.leaves) |*slot, t| {
            slot.* = try t.buffer();
        }

        var output_buffers: [1]zg.Executor.Buffer = undefined;

        for (0..options.warmup_steps) |_| {
            const event = try executor.invoke(
                loaded_program,
                input_buffers,
                &output_buffers,
                .{},
            );
            defer if (event) |completion| executor.release_event(completion);
            // read the result to exercise host transfer before the timed loop
            var loss_tensor = Tensor.from_buffer(executor, output_buffers[0], .f32, &.{});
            errdefer loss_tensor.deinit();
            // `item` synchronizes the transfer.
            _ = try loss_tensor.item(f32);
            loss_tensor.deinit();
        }

        for (0..options.steps) |_| {
            try loop_timer.start_step();
            const event = try executor.invoke(
                loaded_program,
                input_buffers,
                &output_buffers,
                .{},
            );
            defer if (event) |completion| executor.release_event(completion);
            loop_timer.mark("dispatch");

            var loss_tensor = Tensor.from_buffer(executor, output_buffers[0], .f32, &.{});
            errdefer loss_tensor.deinit();
            const loss = try loss_tensor.item(f32);
            loop_timer.mark("sync+read");

            loss_tensor.deinit();
            loop_timer.mark("cleanup");

            loop_timer.end_step(if (options.quiet) null else loss);
        }
    }

    log.info("avg_step_ms={d:.3} (warmup={d} steps={d} sequence_length={d})", .{
        loop_timer.avg_ms(),
        options.warmup_steps,
        options.steps,
        seq,
    });
}

/// Fill each batch row with token IDs and zero padding.
///
/// Each row receives `tokens[0..min(seq, tokens.len)]`.
fn fill_i32_tokens(out: []i32, batch: usize, seq: usize, tokens: []const usize) void {
    @memset(out, 0);
    if (batch == 0 or seq == 0) return;
    for (0..batch) |b| {
        const row = out[b * seq ..][0..seq];
        for (tokens[0..@min(seq, tokens.len)], 0..) |t, i| row[i] = @intCast(t);
    }
}

/// Fill a `[batch, seq]` attention mask with `active_len` active positions.
fn fill_attention_mask(buf: Tensor, batch: usize, seq: usize, active_len: usize) void {
    switch (buf.dtype) {
        inline .f32, .bf16, .f16, .f64 => |tag| {
            const T = tag.StorageType();
            const slice = buf.as_slice(T);
            @memset(slice, tag.encode(f32, 0.0));
            const count = @min(seq, active_len);
            const one = tag.encode(f32, 1.0);
            for (0..batch) |bx| for (0..count) |i| {
                slice[bx * seq + i] = one;
            };
        },
        else => @panic("fill_attention_mask: unsupported dtype"),
    }
}

/// Fill a `[seq, seq]` lower-triangular causal mask.
fn fill_causal_mask(buf: Tensor, seq: usize) void {
    switch (buf.dtype) {
        inline .f32, .bf16, .f16, .f64 => |tag| {
            const T = tag.StorageType();
            const slice = buf.as_slice(T);
            @memset(slice, tag.encode(f32, 0.0));
            const one = tag.encode(f32, 1.0);
            for (0..seq) |i| for (0..i + 1) |j| {
                slice[i * seq + j] = one;
            };
        },
        else => @panic("fill_causal_mask: unsupported dtype"),
    }
}

/// Apply the LLaMA 3 three-region frequency correction in place.
fn llama3_rope_freq_correction(inv_freq: []f32) void {
    const factor: f32 = 32.0;
    const low_freq_factor: f32 = 1.0;
    const high_freq_factor: f32 = 4.0;
    const old_context_len: f32 = 8192.0;

    const low_freq_wavelen = old_context_len / low_freq_factor;
    const high_freq_wavelen = old_context_len / high_freq_factor;

    for (inv_freq) |*freq| {
        const wavelen = 2.0 * std.math.pi / freq.*;
        if (wavelen > low_freq_wavelen) {
            // Low-freq region: divide by factor.
            freq.* /= factor;
        } else if (wavelen >= high_freq_wavelen) {
            // Medium-freq region: smooth interpolation.
            const smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor);
            const scaled = freq.* / factor;
            freq.* = (1.0 - smooth) * scaled + smooth * freq.*;
        }
        // High-freq region (wavelen < high_freq_wavelen): unchanged.
    }
}

/// Precompute RoPE sine and cosine tables with shape `[seq, head_dim / 2]`.
///
/// Inverse frequencies use f32 and a base frequency of 500000.0. The stored
///  values use the destination buffers' dtype.
fn fill_rope_tables(sin: Tensor, cos: Tensor, seq: usize, head_dim_runtime: usize) void {
    const half = head_dim_runtime / 2;

    // supported checkpoints have head dimensions up to 256
    var inv_freq: [128]f32 = undefined;
    std.debug.assert(half <= inv_freq.len);

    const base: f32 = 500000.0;
    for (0..half) |j| {
        const exp = @as(f32, @floatFromInt(2 * j)) / @as(f32, @floatFromInt(head_dim_runtime));
        inv_freq[j] = 1.0 / std.math.pow(f32, base, exp);
    }
    llama3_rope_freq_correction(inv_freq[0..half]);

    switch (sin.dtype) {
        inline .f32, .bf16, .f16, .f64 => |tag| {
            const T = tag.StorageType();
            const s = sin.as_slice(T);
            const c = cos.as_slice(T);
            for (0..seq) |i| for (0..half) |j| {
                const theta = @as(f32, @floatFromInt(i)) * inv_freq[j];
                s[i * half + j] = tag.encode(f32, @sin(theta));
                c[i * half + j] = tag.encode(f32, @cos(theta));
            };
        },
        else => @panic("fill_rope_tables: unsupported dtype"),
    }
}
