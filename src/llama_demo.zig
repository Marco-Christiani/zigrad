const std = @import("std");
const zg = @import("zigrad");
const stz = @import("safetensors_zg");

const llama_model = @import("llama_model.zig");
const Tensor = zg.Tensor;

const log = std.log.scoped(.@"zg/llama-ft-demo");

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
/// This value matches the LLaMA 3.2-1B checkpoint used by the demo.
///
/// TODO(example): Read model dimensions from the checkpoint configuration.
const head_dim: i64 = 64;

/// Concrete llama params type used throughout this demo. Every field path
///  is byte-identical to a safetensors key, so loading is a direct
///  `zg.from_safetensors(LlamaParams, &st, opts)` call.
const LlamaParams = llama_model.LlamaWeights(num_layers);

const BatchSpec = struct {
    x: Tensor,
    target_ids: Tensor,
    attention_mask: Tensor,
    mask: Tensor,
    sin: Tensor,
    cos: Tensor,
};

pub const LlamaKernelProvider = enum {
    // TODO(kernel-provider): Add TVM after the Llama recipe uses the shared store path.
    mirage,
};

pub const LlamaDemoConfig = struct {
    train: bool,
    dtype: zg.DType,
    seq: i64,
    batch: i64 = 1,
    execute_only: bool = false,
    kernel_provider: ?LlamaKernelProvider = null,
};

const upcast_loss = true; // bf16 logits over 128k vocab overflow bf16 range without this

fn loss_fn(params: LlamaParams, batch: BatchSpec) !Tensor {
    return try loss_fn_with_options(params, batch, .{});
}

fn loss_fn_mirage(params: LlamaParams, batch: BatchSpec) !Tensor {
    return try loss_fn_with_options(params, batch, .{ .kernelize_provider = "mirage" });
}

const TrainStepResult = struct { loss_val: Tensor, updated: LlamaParams };

fn train_step_fn(params: LlamaParams, batch: BatchSpec) !TrainStepResult {
    var vg = try zg.transforms.value_and_grad(loss_fn, .{ params, batch });
    defer vg.deinit();
    return try sgd_step(params, &vg, 1e-4);
}

fn train_step_fn_mirage(params: LlamaParams, batch: BatchSpec) !TrainStepResult {
    var vg = try zg.transforms.value_and_grad(loss_fn_mirage, .{ params, batch });
    defer vg.deinit();
    return try sgd_step(params, &vg, 1e-4);
}

fn sgd_step(params: LlamaParams, vg: *zg.transforms.ValueAndGrad, lr: f32) !TrainStepResult {
    var params_tree = try zg.utils.Tree(Tensor).from(vg.grads.allocator, params);
    defer params_tree.deinit();
    const optim = zg.optim.SGD{ .lr = lr };
    var updated = try params_tree.map2(Tensor, &vg.grads, Tensor, optim, zg.optim.SGD.update);
    defer updated.deinit();
    return .{
        .loss_val = vg.value,
        .updated = try updated.extract(LlamaParams),
    };
}

fn loss_fn_with_options(
    params: LlamaParams,
    batch: BatchSpec,
    forward_opts: llama_model.ForwardOptions,
) !Tensor {
    const batch_size: i64 = batch.x.dims()[0];
    const seq: i64 = batch.x.dims()[1];
    const logits = try params.model.forward(
        batch.x,
        batch.mask,
        batch.attention_mask,
        batch.sin,
        batch.cos,
        1e-6,
        forward_opts,
    );
    // Optionally upcast logits to f32 for the loss computation: bf16 logits
    //  over a 128k vocab can overflow the bf16 range during logsumexp.
    const loss_dtype: zg.DType = if (upcast_loss) .f32 else logits.dtype;
    const logits_f = try logits.convert(loss_dtype);
    const attn_mask = try batch.attention_mask.convert(logits_f.dtype);

    // Gather one vocab entry per (batch, seq) row without flattening.
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

    const max_logits = try logits_f.reduce_max(&.{2});
    const max_b = try max_logits.broadcast_in_dim(logits_f.dims(), &.{ 0, 1 });
    const shifted = try logits_f.sub(max_b);

    // JAX-style logsumexp lowering: shift in f32, exp in the model dtype
    //  (bf16 round-trip preserves JAX numerics), accumulate in f32.
    const shifted_f32 = try shifted.convert(.f32);
    const exp_logits = try (try shifted_f32.convert(logits.dtype)).exp();
    const exp_logits_f32 = try exp_logits.convert(.f32);
    const sum_exp = try exp_logits_f32.reduce_sum(&.{2});
    const log_sum = try sum_exp.log();
    const logsumexp = try log_sum.add(try max_logits.convert(log_sum.dtype));

    const loss_per = try logsumexp.sub(try target_logits_2d.convert(logsumexp.dtype));
    const loss_per_out = try loss_per.convert(logits_f.dtype);

    const zero_b = try Tensor.constant_like(attn_mask, 0.0);
    const attn_zero = try attn_mask.compare(zero_b, .{ .direction = .GT, .compare_type = .FLOAT });
    const masked = try loss_per_out.select(attn_zero, zero_b);
    const masked_f = try masked.convert(.f32);
    const loss_sum = try masked_f.reduce_sum(&.{ 0, 1 });

    // Normalize by token count: loss / max(mask.sum(), 1)
    const attn_zero_f = try attn_zero.convert(.f32);
    const mask_count = try attn_zero_f.reduce_sum(&.{ 0, 1 });
    const one = try Tensor.constant_like(mask_count, 1.0);
    const denom = try mask_count.max(one);
    const loss_norm = try loss_sum.div(denom);

    return try loss_norm.convert(logits_f.dtype);
}

pub fn run_llama_ft_demo(
    compilation_context: *zg.compilation.Context,
    pipeline: *zg.compilation.Pipeline,
    environ: *const std.process.Environ.Map,
    warmup_steps: usize,
    steps: usize,
    quiet: bool,
    cfg: LlamaDemoConfig,
) !void {
    const io = compilation_context.io;
    const allocator = compilation_context.allocator;
    const train_mode = cfg.train;
    const model_dtype: zg.DType = cfg.dtype;
    const host_dtype: zg.DType = model_dtype;
    const batch_size: i64 = cfg.batch;
    const execute_only = cfg.execute_only;

    const seq: i64 = cfg.seq;

    // Open weights
    const default_path = "./weights/llama-3.2-1b-instruct/model.safetensors";
    const weights_path: []const u8 = environ.get("ZG_LLAMA_SAFETENSORS_PATH") orelse default_path;

    const mmap_data = zg.utils.mmap_file(weights_path) catch |err| {
        log.err("failed to mmap checkpoint at '{s}': {s}", .{ weights_path, @errorName(err) });
        log.err("Set ZG_LLAMA_SAFETENSORS_PATH to a valid safetensors file, or place one at {s}.", .{default_path});
        return err;
    };
    defer zg.utils.munmap(mmap_data);

    var st_file = try stz.SafeTensorsFile.deserialize(mmap_data, allocator);
    defer st_file.deinit();

    // Load params directly from the checkpoint. Field paths in `LlamaParams` mirror
    //  checkpoint's key hierarchy.
    const host_params = try zg.from_safetensors(LlamaParams, &st_file, .{
        .allocator = allocator,
        .dtype = model_dtype,
    });
    if (!quiet) log.info("Loaded weights from {s}", .{weights_path});

    // Batch host tensors: allocated directly from the known shapes. Sin/cos
    //  tables hold the rotary frequencies for `head_dim/2` channels per
    //  position, so their second dim is derived from `head_dim`.
    const rope_half = @divExact(head_dim, 2);
    const dims_seq_seq: [2]i64 = .{ seq, seq };
    const dims_seq_half: [2]i64 = .{ seq, rope_half };
    const dims_b_s: [2]i64 = .{ batch_size, seq };
    const host_batch = BatchSpec{
        .x = try Tensor.host(.i32, &dims_b_s, .{ .alloc = allocator }),
        .target_ids = try Tensor.host(.i32, &dims_b_s, .{ .alloc = allocator }),
        .attention_mask = try Tensor.host(model_dtype, &dims_b_s, .{ .alloc = allocator }),
        .mask = try Tensor.host(model_dtype, &dims_seq_seq, .{ .alloc = allocator }),
        .sin = try Tensor.host(model_dtype, &dims_seq_half, .{ .alloc = allocator }),
        .cos = try Tensor.host(model_dtype, &dims_seq_half, .{ .alloc = allocator }),
    };

    // Trace reads `.shape`/`.dtype` off the loaded tensors directly (no abstract pass
    //  for concision, but could be done equivalently).
    const inputs_spec = .{ host_params, host_batch };

    // Mirage kernel provider: tune -> store -> pass to compile_cfg.
    const MirageDispatch = if (zg.build_options.has_mirage) zg.mirage.dispatch.MirageDispatchState else void;
    const MirageProviderT = if (zg.build_options.has_mirage) zg.mirage.provider.MirageProvider else void;

    var mirage_dispatch_state: ?MirageDispatch = null;
    defer if (zg.build_options.has_mirage) {
        if (mirage_dispatch_state) |*s| s.deinit();
    };

    var mirage_provider_impl: ?MirageProviderT = null;
    var mirage_providers: [1]zg.pr.kernel.KernelProvider = undefined;

    // TODO(mirage): Connect Mirage selection through either the disconnected
    //  MLIR operation or PR region annotations.
    if (cfg.kernel_provider) |provider| {
        switch (provider) {
            .mirage => {
                if (comptime !zg.build_options.has_mirage) {
                    log.err("mirage provider requested but binary was built without the Mirage integration", .{});
                    return error.MirageUnavailable;
                }
                const mirage_config = zg.mirage.config.Config.from_environ(environ) catch |err| {
                    log.err("Mirage configuration failed: {s}", .{@errorName(err)});
                    return err;
                };
                mirage_dispatch_state = zg.mirage.dispatch.MirageDispatchState.init(
                    allocator,
                    mirage_config.compile,
                );
                mirage_provider_impl = try zg.mirage.provider.MirageProvider.init(
                    &mirage_dispatch_state.?,
                    .{ .runtime = mirage_config.runtime },
                );
                mirage_providers = .{mirage_provider_impl.?.kernel_provider()};

                // TODO(kernel-provider): Connect the Llama recipe to the shared
                //  PR tuning store.
                log.warn("mirage kernel provider not yet supported via store-based path; ignoring", .{});
            },
        }
    }

    const train = zg.train;
    // TODO(kernel-provider): Remove this alternate loss once provider selection
    //  operates on the shared PR recipe.
    const use_mirage_loss = cfg.kernel_provider != null;

    var program = if (train_mode)
        (if (use_mirage_loss)
            try zg.trace(train_step_fn_mirage, allocator, inputs_spec, "llama_ft_step")
        else
            try zg.trace(train_step_fn, allocator, inputs_spec, "llama_ft_step"))
    else
        (if (use_mirage_loss)
            try zg.trace(loss_fn_mirage, allocator, inputs_spec, "llama_ft_step")
        else
            try zg.trace(loss_fn, allocator, inputs_spec, "llama_ft_step"));
    defer program.deinit();
    var exe = try pipeline.run(
        zg.Executor.LoadedProgram,
        &program,
        compilation_context,
    );
    defer exe.deinit();
    const executor = exe.executor;

    const donate = comptime zg.train.donate_argnums(@TypeOf(inputs_spec), &.{0});

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

    // Transfer host -> device.
    var dev_tree = try host_tree.map(Tensor, executor, struct {
        fn f(selected: *zg.Executor, tensor: Tensor) (zg.Executor.Error || error{UnsupportedAval})!Tensor {
            return try tensor.to_device(selected);
        }
    }.f);

    // `dev_tree` deinit is branch-local below: in train-mode we only free
    //  array storage (TrainState owns the leaves) and in inference
    //  mode we use `deinit_with` to free both leaves and storage.

    const loss_dtype: zg.DType = if (upcast_loss) .f32 else host_dtype;

    var loop_timer = zg.utils.LoopTimer{ .io = io, .label = "llama-ft-demo", .quiet = quiet };

    if (train_mode) {
        // TrainState.deinit releases the leaf buffers after donation swaps.
        // This scope releases only the tree arrays and paths.
        defer dev_tree.deinit();

        // Set up state as a convenience for training
        var state = try train.TrainState.init(
            allocator,
            exe,
            dev_tree.leaves,
            program.output_arity("llama_ft_step"),
            .{ .non_donatable_input_indices = donate, .loss_dtype = loss_dtype },
        );
        defer state.deinit(.all);

        for (0..warmup_steps) |_| {
            var result = try state.step();
            // The loss transfer depends on execution, so the execution event
            //  can be released without an explicit wait.
            //
            // TODO(execution): Encapsulate dependent transfer and event release.
            if (result.event) |completion| executor.release_event(completion);
            _ = try result.loss.item(f32);
            result.loss.deinit();
        }

        const nvtx_label: [:0]const u8 = "llama-ft-demo timed loop";
        var nvtx_range = NvtxRange.init() catch null;
        defer if (nvtx_range) |*range| range.deinit();
        if (nvtx_range) |*range| range.push(nvtx_label) catch {};

        for (0..steps) |_| {
            try loop_timer.start_step();
            var result = try state.step();
            // The loss transfer captures the execution dependency.
            //
            // TODO(execution): Encapsulate dependent transfer and event release.
            defer if (result.event) |completion| executor.release_event(completion);
            loop_timer.mark("dispatch");

            // Reading a scalar waits for its transfer.
            const loss: ?f32 = if (quiet or execute_only) null else try result.loss.item(f32);
            loop_timer.mark("sync+read");

            result.loss.deinit();
            loop_timer.mark("cleanup");

            loop_timer.end_step(loss);
        }

        if (nvtx_range) |*range| range.pop() catch {};
    } else {
        // Forward-only mode: simple execute loop, no parameter swapping.
        const fwd_exe = exe;

        defer dev_tree.deinit_with(Tensor.deinit);

        const input_buffers = try allocator.alloc(zg.Executor.Buffer, dev_tree.leaves.len);
        defer allocator.free(input_buffers);
        for (input_buffers, dev_tree.leaves) |*slot, t| {
            slot.* = try t.buffer();
        }

        var output_buffers: [1]zg.Executor.Buffer = undefined;

        for (0..warmup_steps) |_| {
            const event = try executor.invoke(
                fwd_exe,
                input_buffers,
                &output_buffers,
                .{},
            );
            defer if (event) |completion| executor.release_event(completion);
            // Read the result to exercise host transfer before the timed loop.
            var loss_tensor = Tensor.from_buffer(executor, output_buffers[0], loss_dtype, &.{});
            // `item` synchronizes the transfer.
            _ = try loss_tensor.item(f32);
            loss_tensor.deinit();
        }

        const nvtx_label: [:0]const u8 = "llama-ft-demo timed loop";
        var nvtx_range = NvtxRange.init() catch null;
        defer if (nvtx_range) |*range| range.deinit();
        if (nvtx_range) |*range| range.push(nvtx_label) catch {};

        for (0..steps) |_| {
            try loop_timer.start_step();
            const event = try executor.invoke(
                fwd_exe,
                input_buffers,
                &output_buffers,
                .{},
            );
            defer if (event) |completion| executor.release_event(completion);
            loop_timer.mark("dispatch");

            var loss_tensor = Tensor.from_buffer(executor, output_buffers[0], loss_dtype, &.{});
            // .item() implies a sync, so we dont need an explicit barrier
            const loss: ?f32 = if (quiet or execute_only) null else try loss_tensor.item(f32);
            loop_timer.mark("sync+read");

            loss_tensor.deinit();
            loop_timer.mark("cleanup");

            loop_timer.end_step(loss);
        }

        if (nvtx_range) |*range| range.pop() catch {};
    }

    log.info("avg_step_ms={d:.3} (warmup={d} steps={d} seq={d})", .{ loop_timer.avg_ms(), warmup_steps, steps, seq });
    log.info("OK", .{});
}

/// TODO(profiling): Move NVTX integration into a reusable optional module.
const NvtxRange = struct {
    lib: std.DynLib,
    push_fn: *const fn ([*:0]const u8) callconv(.c) c_int,
    pop_fn: *const fn () callconv(.c) c_int,

    pub fn init() !NvtxRange {
        const sonames = [_][:0]const u8{
            "libnvToolsExt.so",
            "libnvToolsExt.so.1",
            "libnvToolsExt.so.1.0",
        };
        for (sonames) |name| {
            if (open_nvtx(name)) |range| return range;
        }
        return error.FileNotFound;
    }

    pub fn deinit(self: *NvtxRange) void {
        self.lib.close();
    }

    pub fn push(self: *NvtxRange, label: [:0]const u8) !void {
        // TODO(profiling): Map NVTX return codes to a specific error set.
        _ = self.push_fn(label);
    }

    pub fn pop(self: *NvtxRange) !void {
        _ = self.pop_fn();
    }
};

fn open_nvtx(path: []const u8) ?NvtxRange {
    if (std.DynLib.open(path)) |lib0| {
        var lib = lib0;
        const push_fn = lib.lookup(*const fn ([*:0]const u8) callconv(.c) c_int, "nvtxRangePushA") orelse {
            lib.close();
            return null;
        };
        const pop_fn = lib.lookup(*const fn () callconv(.c) c_int, "nvtxRangePop") orelse {
            lib.close();
            return null;
        };
        return .{ .lib = lib, .push_fn = push_fn, .pop_fn = pop_fn };
    } else |_| {
        return null;
    }
}

/// Fill an i32 buffer with token IDs, replicated across batches.
///
/// Writes `tokens[0..min(seq, tokens.len)]` into each batch row, padding the remainder with zeros.
/// Handles batch=1 uniformly, we dont special case.
fn fill_i32_tokens(out: []i32, batch: usize, seq: usize, tokens: []const usize) void {
    @memset(out, 0);
    if (batch == 0 or seq == 0) return;
    for (0..batch) |b| {
        const row = out[b * seq ..][0..seq];
        for (tokens[0..@min(seq, tokens.len)], 0..) |t, i| row[i] = @intCast(t);
    }
}

/// Fill a [batch, seq] attn mask: 1.0 for the first `active_len` positions per row, 0.0 elsewhere.
/// Dtype-generic.
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

/// Fill a [seq, seq] lower-triangular causal mask: 1.0 where col <= row, 0.0 above the diagonal.
/// Dtype-generic.
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

/// Apply LLaMA 3 wavelength-based three-region frequency correction in-place.
/// Parameters match LLaMA 3.2-1B config.json rope_scaling section.
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

/// Precompute RoPE sin/cos tables for [seq, head_dim/2].
///
/// Inverse frequencies are computed in f32 (precision requirement), then encoded into the buffer's
///  dtype. Applies LLaMA 3 wavelength-based frequency correction via `llama3_rope_freq_correction`.
/// Base frequency is 500000.0 (LLaMA 3.2-1B config).
fn fill_rope_tables(sin: Tensor, cos: Tensor, seq: usize, head_dim_runtime: usize) void {
    const half = head_dim_runtime / 2;

    // stack buffer sized for any llama variant we expect (head_dim <= 256)
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
