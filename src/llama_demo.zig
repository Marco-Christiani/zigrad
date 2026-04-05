const std = @import("std");
const zg = @import("zigrad");
const stz = @import("safetensors_zg");

const llama_model = @import("llama_model.zig");
const Tensor = zg.Tensor;

const log = std.log.scoped(.@"zg/llama-ft-demo");

/// Create an iota tensor from a traced tensor's builder.
/// TODO: missing method?
fn iota_from(t: Tensor, out_dtype: zg.DType, out_dims: []const i64, iota_dim: i64) !Tensor {
    const b = t.backing.traced.builder;
    return Tensor.from_var(b, try b.iota(out_dtype, out_dims, iota_dim));
}

const num_layers: usize = 16;

// TODO: these arent necessary, should directly use the llama model struct(s)
const LayerSpec = struct {
    input_norm: Tensor,
    post_norm: Tensor,
    qkv_proj: Tensor,
    o_proj: Tensor,
    gate_proj: Tensor,
    up_proj: Tensor,
    down_proj: Tensor,
};

const ParamsSpec = struct {
    w_emb: Tensor,
    w_out: Tensor,
    norm: Tensor,
    layers: [num_layers]LayerSpec,
};

const BatchSpec = struct {
    x: Tensor,
    target_ids: Tensor,
    attention_mask: Tensor,
    mask: Tensor,
    sin: Tensor,
    cos: Tensor,
};

pub const LlamaKernelProvider = enum {
    // TODO: add tvm kp
    mirage,
};

pub const LlamaDemoConfig = struct {
    train: bool,
    dtype: zg.DType,
    seq: i64,
    batch: i64 = 1,
    canonical_shapes: bool = false,
    execute_only: bool = false,
    kernel_provider: ?LlamaKernelProvider = null,
};

const upcast_loss = true; // bf16 logits over 128k vocab overflow bf16 range without this

fn loss_fn(params: ParamsSpec, batch: BatchSpec) !Tensor {
    return loss_fn_with_options(params, batch, .{});
}

fn loss_fn_mirage(params: ParamsSpec, batch: BatchSpec) !Tensor {
    return loss_fn_with_options(params, batch, .{ .kernelize_provider = "mirage" });
}

const TrainStepResult = struct { loss_val: Tensor, updated: ParamsSpec };

fn train_step_fn(params: ParamsSpec, batch: BatchSpec) !TrainStepResult {
    var vg = try zg.frontend.transforms.value_and_grad(loss_fn, .{ params, batch });
    defer vg.deinit();
    return sgd_step(params, &vg, 1e-4);
}

fn train_step_fn_mirage(params: ParamsSpec, batch: BatchSpec) !TrainStepResult {
    var vg = try zg.frontend.transforms.value_and_grad(loss_fn_mirage, .{ params, batch });
    defer vg.deinit();
    return sgd_step(params, &vg, 1e-4);
}

fn sgd_step(params: ParamsSpec, vg: *zg.frontend.transforms.ValueAndGrad, lr: f32) !TrainStepResult {
    var params_tree = try zg.utils.Tree(Tensor).from(vg.grads.allocator, params);
    defer params_tree.deinit();
    const optim = zg.frontend.optim.SGD{ .lr = lr };
    var updated = try params_tree.map2(Tensor, &vg.grads, Tensor, optim, zg.frontend.optim.SGD.update);
    defer updated.deinit();
    return .{
        .loss_val = vg.value,
        .updated = updated.extract(ParamsSpec),
    };
}

fn loss_fn_with_options(
    params: ParamsSpec,
    batch: BatchSpec,
    forward_opts: llama_model.ForwardOptions,
) !Tensor {
    const batch_size: i64 = batch.x.dims()[0];
    const seq: i64 = batch.x.dims()[1];
    var layers: [num_layers]llama_model.LayerWeights = undefined;
    inline for (0..num_layers) |idx| {
        const p = params.layers[idx];
        layers[idx] = .{
            .input_norm = p.input_norm,
            .post_norm = p.post_norm,
            .qkv_proj = p.qkv_proj,
            .o_proj = p.o_proj,
            .gate_proj = p.gate_proj,
            .up_proj = p.up_proj,
            .down_proj = p.down_proj,
        };
    }
    const logits = try llama_model.forward(batch.x, batch.mask, batch.attention_mask, batch.sin, batch.cos, .{
        .w_emb = params.w_emb,
        .w_out = params.w_out,
        .norm = params.norm,
        .layers = layers[0..],
    }, 1e-6, forward_opts);
    const logits_f0 = if (upcast_loss and logits.dtype == .bf16) try logits.convert(.f32) else logits;
    const logits_f = logits_f0;
    const attn_mask0 = if (batch.attention_mask.dtype == logits_f.dtype)
        batch.attention_mask
    else
        try batch.attention_mask.convert(logits_f.dtype);
    const attn_mask = attn_mask0;

    // Gather one vocab entry per (batch, seq) row without flattening.
    const row_b = try iota_from(logits_f, .i32, &.{ batch_size, seq }, 0);
    const row_s = try iota_from(logits_f, .i32, &.{ batch_size, seq }, 1);
    const tgt_ids = if (batch.target_ids.dtype == .i32)
        batch.target_ids
    else
        try batch.target_ids.convert(.i32);
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

    // Match JAX-style logsumexp lowering: subtract in f32, exp in bf16 (when model dtype is bf16),
    // then accumulate reductions in f32.
    const shifted_f32 = if (shifted.dtype == .f32) shifted else try shifted.convert(.f32);
    const exp_in = if (logits.dtype == .bf16) try shifted_f32.convert(.bf16) else shifted_f32;
    const exp_logits = try exp_in.exp();
    const exp_logits_f32 = if (exp_logits.dtype == .f32) exp_logits else try exp_logits.convert(.f32);
    const sum_exp = try exp_logits_f32.reduce_sum(&.{2});
    const log_sum = try sum_exp.log();
    const max_f = if (max_logits.dtype == log_sum.dtype)
        max_logits
    else
        try max_logits.convert(log_sum.dtype);
    const logsumexp = try log_sum.add(max_f);

    const target_f = if (target_logits_2d.dtype == logsumexp.dtype)
        target_logits_2d
    else
        try target_logits_2d.convert(logsumexp.dtype);
    const loss_per = try logsumexp.sub(target_f);
    const loss_per_out = if (loss_per.dtype == logits_f.dtype)
        loss_per
    else
        try loss_per.convert(logits_f.dtype);

    const zero_b = try Tensor.constant_like(attn_mask, 0.0);
    const attn_zero = try attn_mask.compare(zero_b, .{ .direction = .GT, .compare_type = .FLOAT });
    const masked = try loss_per_out.select(attn_zero, zero_b);
    const masked_f = if (masked.dtype == .f32) masked else try masked.convert(.f32);
    const loss_sum = try masked_f.reduce_sum(&.{ 0, 1 });
    if (loss_sum.dtype == logits_f.dtype) return loss_sum;
    return loss_sum.convert(logits_f.dtype);
}

pub fn run_llama_ft_demo(
    allocator: std.mem.Allocator,
    b: *zg.Backend,
    device: zg.Backend.Device,
    dump_pr: ?*zg.pipeline.DumpConfig,
    dump_mlir: ?*zg.pipeline.DumpConfig,
    dump_optimized: ?*zg.pipeline.DumpConfig,
    warmup_steps: usize,
    steps: usize,
    quiet: bool,
    cfg: LlamaDemoConfig,
    dump_kernels: bool,
) !void {
    const train_mode = cfg.train;
    const model_dtype: zg.DType = cfg.dtype;
    const host_dtype: zg.DType = model_dtype;
    const batch_size: i64 = cfg.batch;
    const execute_only = cfg.execute_only;

    const seq: i64 = cfg.seq;
    const vocab: i64 = if (cfg.canonical_shapes) 4096 else 128256;
    const hidden: i64 = if (cfg.canonical_shapes) 512 else 2048;
    const kv_out: i64 = 512;
    const mlp_hidden: i64 = hidden * 4;
    const qkv_out: i64 = hidden + kv_out + kv_out;

    const donatable: Tensor.AbstractOpts = .{ .donatable = true };
    var layers_spec: [num_layers]LayerSpec = undefined;
    inline for (0..num_layers) |i| {
        layers_spec[i] = .{
            .input_norm = Tensor.abstract(model_dtype, &.{hidden}, donatable),
            .post_norm = Tensor.abstract(model_dtype, &.{hidden}, donatable),
            .qkv_proj = Tensor.abstract(model_dtype, &.{ hidden, qkv_out }, donatable),
            .o_proj = Tensor.abstract(model_dtype, &.{ hidden, hidden }, donatable),
            .gate_proj = Tensor.abstract(model_dtype, &.{ hidden, mlp_hidden }, donatable),
            .up_proj = Tensor.abstract(model_dtype, &.{ hidden, mlp_hidden }, donatable),
            .down_proj = Tensor.abstract(model_dtype, &.{ mlp_hidden, hidden }, donatable),
        };
    }

    const params_spec = ParamsSpec{
        .w_emb = Tensor.abstract(model_dtype, &.{ vocab, hidden }, donatable),
        .w_out = Tensor.abstract(model_dtype, &.{ hidden, vocab }, donatable),
        .norm = Tensor.abstract(model_dtype, &.{hidden}, donatable),
        .layers = layers_spec,
    };
    var dims_seq_seq: [2]i64 = .{ seq, seq };
    var dims_seq_32: [2]i64 = .{ seq, 32 };
    var dims_b_s: [2]i64 = .{ batch_size, seq };
    const batch_spec = BatchSpec{
        .x = Tensor.abstract(.i32, dims_b_s[0..], .{}),
        .target_ids = Tensor.abstract(.i32, dims_b_s[0..], .{}),
        .attention_mask = Tensor.abstract(model_dtype, dims_b_s[0..], .{}),
        .mask = Tensor.abstract(model_dtype, dims_seq_seq[0..], .{}),
        .sin = Tensor.abstract(model_dtype, dims_seq_32[0..], .{}),
        .cos = Tensor.abstract(model_dtype, dims_seq_32[0..], .{}),
    };
    const inputs_spec = .{ params_spec, batch_spec };

    var compile_cfg = zg.frontend.CompileConfig{
        .entry_name = "llama_ft_step",
        .dump_pr = if (dump_pr) |dump_cfg| dump_cfg.* else null,
        .dump_mlir = if (dump_mlir) |dump_cfg| dump_cfg.* else null,
        .dump_optimized = if (dump_optimized) |dump_cfg| dump_cfg.* else null,
        .dump_kernels = dump_kernels,
    };

    if (compile_cfg.dump_mlir != null) {
        compile_cfg.lower.encoding = .text;
    }

    // Mirage kernel provider: tune -> store -> pass to compile_cfg.
    const MirageDispatch = if (zg.build_options.has_mirage) zg.mirage.dispatch.MirageDispatchState else void;
    const MirageProviderT = if (zg.build_options.has_mirage) zg.mirage.provider.MirageProvider else void;

    var mirage_dispatch_state: ?MirageDispatch = null;
    defer if (zg.build_options.has_mirage) {
        if (mirage_dispatch_state) |*s| s.deinit();
    };

    var mirage_provider_impl: ?MirageProviderT = null;
    var mirage_providers: [1]zg.kernel.KernelProvider = undefined;

    // TODO: Mirage currently only works via MLIR-level patterns (select pass),
    // which is not wired into compile_program. Store-based PR-level Mirage
    // kernelization requires region annotations in the frontend model.
    // For now, the mirage provider path is disabled until MLIR pipeline
    // assembly is supported or PR-level annotations are added.
    if (cfg.kernel_provider) |provider| {
        switch (provider) {
            .mirage => {
                if (comptime !zg.build_options.has_mirage) {
                    log.err("mirage provider requested but binary was built without mirage support (headers not found in SDK)", .{});
                    return error.MirageUnavailable;
                }
                mirage_dispatch_state = try zg.mirage.dispatch.MirageDispatchState.init(allocator);
                mirage_provider_impl = .{
                    .allocator = allocator,
                    .dispatch_state = &mirage_dispatch_state.?,
                };
                mirage_providers = .{mirage_provider_impl.?.kernel_provider()};

                // Mirage uses MLIR-level pattern matching, not PR-level store.
                // This path is a placeholder -- full MLIR pipeline assembly
                // will be added in a future phase.
                log.warn("mirage kernel provider not yet supported via store-based path; ignoring", .{});
            },
        }
    }

    const train = zg.frontend.train;
    // TODO: Fix this later when KP starts stabilizing
    const use_mirage_loss = cfg.kernel_provider != null;

    var compiled_train: ?zg.frontend.CompiledModel = null;
    var compiled_fwd: ?zg.frontend.CompiledModel = null;
    if (train_mode) {
        compiled_train = if (use_mirage_loss)
            try zg.frontend.compile(train_step_fn_mirage, allocator, b, device, inputs_spec, compile_cfg)
        else
            try zg.frontend.compile(train_step_fn, allocator, b, device, inputs_spec, compile_cfg);
    } else {
        compiled_fwd = if (use_mirage_loss)
            try zg.frontend.compile(loss_fn_mirage, allocator, b, device, inputs_spec, compile_cfg)
        else
            try zg.frontend.compile(loss_fn, allocator, b, device, inputs_spec, compile_cfg);
    }
    defer {
        if (compiled_train) |*ct| {
            b.deinit_executable(ct.exe);
            ct.deinit();
        }
        if (compiled_fwd) |*cf| {
            b.deinit_executable(cf.exe);
            cf.deinit();
        }
    }

    // Build host buffers from spec tree - shapes and dtypes derived from specs.
    var spec_tree = try zg.utils.Tree(Tensor).from(allocator, inputs_spec);
    defer spec_tree.deinit();

    var host_tree = try spec_tree.map(zg.HostBuffer, allocator, struct {
        fn f(alloc: std.mem.Allocator, spec: Tensor) anyerror!zg.HostBuffer {
            return zg.HostBuffer.init(alloc, spec.shape, spec.dtype);
        }
    }.f);
    defer host_tree.deinit_with(zg.HostBuffer.deinit);

    // Load or synthesize weights into param (donatable) leaves
    const default_path = "./weights/llama-3.2-1b-instruct/model.safetensors";
    const weights_path = std.process.getEnvVarOwned(allocator, "ZG_LLAMA_SAFETENSORS_PATH") catch default_path;
    defer if (!std.mem.eql(u8, weights_path, default_path)) allocator.free(weights_path);

    const donatable_mask = if (compiled_train) |ct| ct.donatable else compiled_fwd.?.donatable;
    var load_result = try load_llama_weights(allocator, weights_path, &host_tree, .{
        .hidden = hidden,
        .kv_out = kv_out,
    });
    // mmap must stay alive until after device upload (borrowed buffers reference it)
    defer if (load_result) |*lr| lr.deinit();

    if (load_result != null) {
        if (!quiet) {
            log.info("Loaded weights from {s}", .{weights_path});
        }
    } else {
        // Fill param leaves with synthetic pattern
        for (host_tree.leaves, donatable_mask) |*buf, is_donatable| {
            if (!is_donatable) continue;
            fill_pattern(buf, 1e-3, 0.0);
        }
        log.warn("Using synthetic weights (set ZG_LLAMA_SAFETENSORS_PATH to use a specific checkpoint)", .{});
    }

    // Fill batch leaves
    const token_seed = [_]usize{ 128000, 128009, 128001, 128008 };
    const target_seed = [_]usize{ 128009, 128001, 128008, 128001 };
    var tokens: [token_seed.len]usize = undefined;
    var targets: [target_seed.len]usize = undefined;
    const vocab_usize: usize = @intCast(vocab);
    for (token_seed, 0..) |value, idx| {
        tokens[idx] = value % vocab_usize;
    }
    for (target_seed, 0..) |value, idx| {
        targets[idx] = value % vocab_usize;
    }
    const batch_usize: usize = @intCast(batch_size);
    const seq_usize: usize = @intCast(seq);

    const host_x = host_tree.get("1.x") orelse return error.MissingSpec;
    const host_target_ids = host_tree.get("1.target_ids") orelse return error.MissingSpec;
    const host_attention_mask = host_tree.get("1.attention_mask") orelse return error.MissingSpec;
    const host_mask = host_tree.get("1.mask") orelse return error.MissingSpec;
    const host_sin = host_tree.get("1.sin") orelse return error.MissingSpec;
    const host_cos = host_tree.get("1.cos") orelse return error.MissingSpec;

    fill_i32_tokens(host_x.as_slice(i32), batch_usize, seq_usize, &tokens);
    fill_i32_tokens(host_target_ids.as_slice(i32), batch_usize, seq_usize, &targets);
    fill_attention_mask(host_attention_mask, batch_usize, seq_usize, tokens.len);
    fill_causal_mask(host_mask, seq_usize);
    fill_rope_tables(host_sin, host_cos, seq_usize, 64);

    // Transfer host -> device.
    var dev_tree = try b.transfer(device, &host_tree, .to_device);
    defer dev_tree.deinit(); // array only. buffer ownership managed by TrainState / defer below

    const loss_dtype: zg.DType = if (upcast_loss) .f32 else host_dtype;
    var loss_host = try zg.HostBuffer.init(allocator, .{}, loss_dtype);
    defer loss_host.deinit();

    var loop_timer = zg.utils.LoopTimer{ .label = "llama-ft-demo", .quiet = quiet };

    if (train_mode) {
        // Set up state as a convenience for training
        var state = try train.TrainState.init_from_model(
            allocator,
            &compiled_train.?,
            b,
            dev_tree.leaves,
        );
        defer state.deinit(.all);

        for (0..warmup_steps) |_| {
            const result = try state.step();
            // Do not await the execution event as buffer_to_host internally chains behind the execution.
            // Also, deinit only the event handle.
            // TODO: this expose a bit of an annoying aspect of the API
            if (result.event) |ev| b.deinit_event(ev);
            _ = try read_loss(b, result.loss_buf, &loss_host, loss_dtype);
            b.deinit_buffer(result.loss_buf);
        }

        const nvtx_label: [:0]const u8 = "llama-ft-demo timed loop";
        var nvtx_range = NvtxRange.init() catch null;
        defer if (nvtx_range) |*range| range.deinit();
        if (nvtx_range) |*range| range.push(nvtx_label) catch {};

        for (0..steps) |_| {
            try loop_timer.start_step();
            const result = try state.step();
            // deinit execution event without awaiting, the transfer event from buffer_to_host
            //  captures the full dependency.
            // TODO: this expose a bit of an annoying aspect of the API
            if (result.event) |ev| b.deinit_event(ev);
            loop_timer.mark("dispatch");

            const loss: ?f32 = if (quiet or execute_only) null else try read_loss(b, result.loss_buf, &loss_host, loss_dtype);
            loop_timer.mark("sync+read");

            b.deinit_buffer(result.loss_buf);
            loop_timer.mark("cleanup");

            loop_timer.end_step(loss);
        }

        if (nvtx_range) |*range| range.pop() catch {};
    } else {
        // Forward-only mode: simple execute loop, no parameter swapping.
        const fwd_exe = compiled_fwd.?.exe;

        defer for (dev_tree.leaves) |buf| b.deinit_buffer(buf);

        var output_bufs: [1]zg.Backend.Buffer = undefined;

        for (0..warmup_steps) |_| {
            const ev = try b.execute_into(fwd_exe, dev_tree.leaves, &output_bufs, null, .{});
            if (ev) |e| b.deinit_event(e);
            // read to warm up the DMA path otherwise first timed step pays a ~65ms lazy-init penalty
            _ = try read_loss(b, output_bufs[0], &loss_host, loss_dtype);
            b.deinit_buffer(output_bufs[0]);
        }

        const nvtx_label: [:0]const u8 = "llama-ft-demo timed loop";
        var nvtx_range = NvtxRange.init() catch null;
        defer if (nvtx_range) |*range| range.deinit();
        if (nvtx_range) |*range| range.push(nvtx_label) catch {};

        for (0..steps) |_| {
            try loop_timer.start_step();
            const event = try b.execute_into(fwd_exe, dev_tree.leaves, &output_bufs, null, .{});
            if (event) |ev| b.deinit_event(ev);
            loop_timer.mark("dispatch");

            const loss: ?f32 = if (quiet or execute_only) null else try read_loss(b, output_bufs[0], &loss_host, loss_dtype);
            loop_timer.mark("sync+read");

            b.deinit_buffer(output_bufs[0]);
            loop_timer.mark("cleanup");

            loop_timer.end_step(loss);
        }

        if (nvtx_range) |*range| range.pop() catch {};
    }

    log.info("avg_step_ms={d:.3} (warmup={d} steps={d} seq={d})", .{ loop_timer.avg_ms(), warmup_steps, steps, seq });
    log.info("OK", .{});
}

/// TODO: this doesnt really belong here
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
        // TODO: when this is moved and built out, need proper error checking/mapping
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

/// Load llama weights from safetensors into host buffers via tree path lookup.
///
/// Buffers are located by their tree path (e.g. `"0.w_emb"`, `"0.layers.3.qkv_proj"`),
///  so field ordering in `ParamsSpec` is irrelevant. Shapes are derived from the
///  buffers. `model_dims` provides extra dimensions for QKV projection splitting.
const ModelDims = struct {
    hidden: i64,
    kv_out: i64,
};

/// Result of loading weights from a safetensors file.
///
/// When zero-copy is used, some host_tree leaves are borrowed views into the
///  mmap'd file data. The caller must keep `mmap_data` alive until those
///  buffers are uploaded to device, then munmap.
/// TODO: this is poorly named and could be implemented in method form instead,
///  it would be ideal if we could unify the optionality (mmap, direct, etc).
const LoadResult = struct {
    mmap_data: ?[]align(std.heap.page_size_min) u8,

    pub fn deinit(self: *LoadResult) void {
        if (self.mmap_data) |m| std.posix.munmap(m);
    }
};

fn load_llama_weights(
    allocator: std.mem.Allocator,
    path: []const u8,
    host_tree: *zg.utils.Tree(zg.HostBuffer),
    model_dims: ModelDims,
) !?LoadResult {
    const data = mmap_file(path) catch return null;
    errdefer std.posix.munmap(data);

    var st_file = try stz.SafeTensorsFile.deserialize(data, allocator);
    defer st_file.deinit();

    const get_buf = struct {
        fn f(tree: *zg.utils.Tree(zg.HostBuffer), tree_path: []const u8) !*zg.HostBuffer {
            return tree.get(tree_path) orelse error.MissingSpec;
        }
    }.f;

    // top-level params
    const w_emb = try get_buf(host_tree, "0.w_emb");
    try load_weight(try st_file.get("model.embed_tokens.weight"), w_emb, .direct);

    try load_weight(try st_file.get("model.norm.weight"), try get_buf(host_tree, "0.norm"), .direct);

    // per-layer weights
    const hidden_u: usize = @intCast(model_dims.hidden);
    const kv_out_u: usize = @intCast(model_dims.kv_out);

    inline for (0..num_layers) |i| {
        const st_prefix = comptime std.fmt.comptimePrint("model.layers.{d}.", .{i});
        const tree_prefix = comptime std.fmt.comptimePrint("0.layers.{d}.", .{i});

        try load_weight(try st_file.get(st_prefix ++ "input_layernorm.weight"), try get_buf(host_tree, tree_prefix ++ "input_norm"), .direct);
        try load_weight(try st_file.get(st_prefix ++ "post_attention_layernorm.weight"), try get_buf(host_tree, tree_prefix ++ "post_norm"), .direct);

        // qkv_proj: concatenated from q, k, v with transposition
        const qkv = try get_buf(host_tree, tree_prefix ++ "qkv_proj");
        try load_weight(try st_file.get(st_prefix ++ "self_attn.q_proj.weight"), qkv, .{ .transposed_into_cols = .{ .col_offset = 0 } });
        try load_weight(try st_file.get(st_prefix ++ "self_attn.k_proj.weight"), qkv, .{ .transposed_into_cols = .{ .col_offset = hidden_u } });
        try load_weight(try st_file.get(st_prefix ++ "self_attn.v_proj.weight"), qkv, .{ .transposed_into_cols = .{ .col_offset = hidden_u + kv_out_u } });

        try load_weight(try st_file.get(st_prefix ++ "self_attn.o_proj.weight"), try get_buf(host_tree, tree_prefix ++ "o_proj"), .transposed);
        try load_weight(try st_file.get(st_prefix ++ "mlp.gate_proj.weight"), try get_buf(host_tree, tree_prefix ++ "gate_proj"), .transposed);
        try load_weight(try st_file.get(st_prefix ++ "mlp.up_proj.weight"), try get_buf(host_tree, tree_prefix ++ "up_proj"), .transposed);
        try load_weight(try st_file.get(st_prefix ++ "mlp.down_proj.weight"), try get_buf(host_tree, tree_prefix ++ "down_proj"), .transposed);
    }

    // w_out: use lm_head.weight if present, otherwise transpose w_emb.
    const w_out = try get_buf(host_tree, "0.w_out");
    const lm_head_w_name = "lm_head.weight";
    const lm_view = st_file.get(lm_head_w_name) catch |err| switch (err) {
        stz.Error.TensorNotFound => null,
        else => return err,
    };
    if (lm_view) |view| {
        try load_weight(view, w_out, .transposed);
    } else {
        log.warn("{s} not found in checkpoint, assuming tied weights", .{lm_head_w_name});
        try transpose_buf(w_emb, w_out);
    }

    return .{ .mmap_data = data };
}

/// Load a single weight tensor from a safetensors view into a host buffer.
///
/// For `.direct` layout with matching dtype, replaces the heap-backed buffer
///  with a zero-copy borrowed view into the mmap'd file data. For transposed
///  or cross-dtype loads, copies element-by-element into the existing buffer.
/// TODO: Seems generally useful. consider moving into stz or zigrad libs and make
///  dtype-generic (comptime dtype). Actually would be rather interesting to explore
///  a backend JIT path here, these are simple transforms that can be expressed by
///  all backends.
fn load_weight(view: stz.TensorView, buf: *zg.HostBuffer, layout: CopyLayout) !void {
    switch (layout) {
        .direct => {
            if (!shape_eql(view.info.shape, buf.shape.const_slice())) return error.TensorShapeMismatch;

            // zero-copy: replace the heap buffer with a borrowed view
            if (stz_dtype_matches(view.info.dtype, buf.dtype)) {
                buf.deinit(); // free the pre-allocated heap buffer
                buf.* = zg.HostBuffer.borrow(view.data, buf.shape, buf.dtype);
                return;
            }

            // cross-dtype: element-wise copy into existing buffer
            const src_count = buf.shape.num_elements();
            var i: usize = 0;
            while (i < src_count) : (i += 1) {
                write_element(buf, i, read_element(view, i));
            }
        },
        .transposed, .transposed_into_cols => {
            try copy_view_to_buf(view, buf, layout);
        },
    }
}

/// Check if a safetensors dtype matches a zigrad dtype.
fn stz_dtype_matches(stz_dt: stz.Dtype, zg_dt: zg.DType) bool {
    return switch (zg_dt) {
        .f32 => stz_dt == .f32,
        .bf16 => stz_dt == .bf16,
        .f16 => stz_dt == .f16,
        .f64 => stz_dt == .f64,
        .i32 => stz_dt == .i32,
        .i64 => stz_dt == .i64,
        inline else => |x| @panic("Unsupported dtype " ++ @tagName(x)),
    };
}

/// Layout modes for weight loading from safetensors into host buffers.
///
/// `.direct` is handled inline by `load_weight` (zero-copy borrow when dtype
/// matches, or element-wise cross-dtype copy).
/// `.transposed` and `.transposed_into_cols` are handled by `copy_view_to_buf`.
const CopyLayout = union(enum) {
    /// 1:1 copy. No shape transformation. Handled by `load_weight` directly.
    direct,
    /// 2D matrix transpose: source [R, C] -> dst [C, R].
    transposed,
    /// Transpose a sub-matrix into a column slice of a wider destination.
    /// Used for concatenating Q/K/V projections into a single buffer.
    transposed_into_cols: struct { col_offset: usize },
};

/// Copy a safetensors view into a host buffer with layout transformation.
///
/// Reads element-by-element via `read_element`/`write_element`, converting dtype and applying the
///  requested transpose one pass with no intermediate buffers.
/// No intermediate buffer is allocated. Only handles `.transposed` and `.transposed_into_cols`,
///  `.direct` is handled by `load_weight`.
/// TODO: inline this.
fn copy_view_to_buf(view: stz.TensorView, dst: *zg.HostBuffer, layout: CopyLayout) !void {
    const src_rows = view.info.shape[0];
    const src_cols = if (view.info.shape.len >= 2) view.info.shape[1] else 1;

    switch (layout) {
        .direct => @panic("should be unreachable. should never copy_view_to_buf on direct tensor"), // handled inline by load_weight
        .transposed => {
            if (view.info.shape.len != 2 or dst.shape.const_slice().len != 2) return error.TensorShapeMismatch;
            const dst_rows: usize = @intCast(dst.shape.const_slice()[0]);
            const dst_cols: usize = @intCast(dst.shape.const_slice()[1]);
            if (dst_rows != src_cols or dst_cols != src_rows) return error.TensorShapeMismatch;

            var r: usize = 0;
            while (r < src_rows) : (r += 1) {
                var c: usize = 0;
                while (c < src_cols) : (c += 1) {
                    write_element(dst, c * dst_cols + r, read_element(view, r * src_cols + c));
                }
            }
        },
        .transposed_into_cols => |opts| {
            if (view.info.shape.len != 2 or dst.shape.const_slice().len != 2) return error.TensorShapeMismatch;
            const dst_stride: usize = @intCast(dst.shape.const_slice()[1]);
            // source is [out_dim, in_dim] (row-major), dst column slice is [in_dim, out_dim] at col_offset.
            const in_dim = src_cols;
            const out_dim = src_rows;
            if (opts.col_offset + out_dim > dst_stride) return error.TensorShapeMismatch;

            var o: usize = 0;
            while (o < out_dim) : (o += 1) {
                var i: usize = 0;
                while (i < in_dim) : (i += 1) {
                    write_element(dst, i * dst_stride + (opts.col_offset + o), read_element(view, o * in_dim + i));
                }
            }
        },
    }
}

/// Read one element from a safetensors view as f32.
/// TODO: Seems generally useful. consider moving into stz or zigrad libs and make
///  dtype-generic (comptime dtype).
inline fn read_element(view: stz.TensorView, idx: usize) f32 {
    return switch (view.info.dtype) {
        .f32 => std.mem.bytesAsSlice(f32, view.data)[idx],
        .bf16 => zg.DType.bf16.decode_f32(std.mem.bytesAsSlice(u16, view.data)[idx]),
        else => unreachable,
    };
}

/// Write one f32 element into a host buffer, converting to the buffer's dtype.
/// TODO: Seems generally useful. consider moving into stz or zigrad libs and make
///  dtype-generic (comptime dtype).
inline fn write_element(buf: *zg.HostBuffer, idx: usize, val: f32) void {
    switch (buf.dtype) {
        .f32 => buf.as_slice(f32)[idx] = val,
        .bf16 => buf.as_slice(u16)[idx] = zg.DType.bf16.encode_f32(val),
        else => unreachable,
    }
}

/// Transpose src [rows, cols] into dst [cols, rows] through raw byte copies.
///
/// Used for the w_emb -> w_out fallback path where both tensors are already in host buffers
///  (not safetensors views).
/// Dtype-agnostic: copies `dtype.size_in_bytes()` bytes per element, so works for any dtype without
///  per-type branches. Both buffers must have the same dtype.
/// TODO: Consider a method on HostBuffer for basic ops like this, could provide naive native and
///  blas impls. Might make sense to consider (re-)using a high level frontend API that computes
///  this using the backend (simple JIT path).
fn transpose_buf(src: *zg.HostBuffer, dst: *zg.HostBuffer) !void {
    if (src.shape.const_slice().len != 2 or dst.shape.const_slice().len != 2) return error.TensorShapeMismatch;
    const src_rows: usize = @intCast(src.shape.const_slice()[0]);
    const src_cols: usize = @intCast(src.shape.const_slice()[1]);
    if (@as(usize, @intCast(dst.shape.const_slice()[0])) != src_cols or @as(usize, @intCast(dst.shape.const_slice()[1])) != src_rows)
        return error.TensorShapeMismatch;
    if (src.dtype != dst.dtype) return error.TensorDtypeMismatch;

    const elem = src.dtype.size_in_bytes();
    const s = src.data();
    const d = dst.data_mut();
    for (0..src_rows) |r| {
        for (0..src_cols) |c| {
            const src_off = (r * src_cols + c) * elem;
            const dst_off = (c * src_rows + r) * elem;
            @memcpy(d[dst_off..][0..elem], s[src_off..][0..elem]);
        }
    }
}

/// Memory-map a file read-only.
///
/// Returns a page-aligned slice backed by the kernel page cache.
/// Caller must `std.posix.munmap` when done.
/// TODO: This is duplicated (eg in llama demo). consider moving into stz or zigrad libs.
fn mmap_file(path: []const u8) ![]align(std.heap.page_size_min) u8 {
    var file = if (std.fs.path.isAbsolute(path))
        try std.fs.openFileAbsolute(path, .{})
    else
        try std.fs.cwd().openFile(path, .{});
    defer file.close();

    const stat = try file.stat();
    const size: usize = @intCast(stat.size);

    return std.posix.mmap(
        null,
        size,
        std.posix.PROT.READ,
        .{ .TYPE = .SHARED },
        file.handle,
        0,
    );
}

/// Compare a safetensors shape (usize) with an expected shape (i64).
/// TODO: Seems generally useful. consider moving into stz or zigrad libs.
fn shape_eql(stz_shape: []const usize, expected: []const i64) bool {
    if (stz_shape.len != expected.len) return false;
    for (stz_shape, expected) |a, b| {
        if (a != @as(usize, @intCast(b))) return false;
    }
    return true;
}

/// Read the scalar loss value from a device buffer, returning f32.
///
/// Copies `loss_buf` into the caller-provided `loss_host` staging buffer, awaits the transfer event,
///  then decodes the first element from the buffer's dtype to f32.
///
/// Does not take ownership of `loss_buf`.
/// TODO: Seems like generally useful boilerplate that may belong in Tensor, Buffer, HostBuffer, or similar
///  like a .item() method.
fn read_loss(b: *zg.Backend, loss_buf: zg.Backend.Buffer, loss_host: *zg.HostBuffer, loss_dtype: zg.DType) !f32 {
    if (try b.buffer_to_host(loss_buf, loss_host.data_mut())) |ev| {
        try b.await_event(ev);
        b.deinit_event(ev);
    }
    return switch (loss_dtype) {
        inline .f32, .bf16, .f16, .f64 => |tag| tag.decode_f32(std.mem.bytesAsSlice(tag.StorageType(), loss_host.data())[0]),
        else => @panic("read_loss: unsupported dtype"),
    };
}

/// Fill a host buffer with a deterministic ramp pattern: `offset + scale * (i % 1024)`.
///
/// Dtype-generic via `inline switch` + `DType.encode_f32`.
/// Supports any float dtype (f32, bf16, f16, f64).
fn fill_pattern(buf: *zg.HostBuffer, scale: f32, offset: f32) void {
    switch (buf.dtype) {
        inline .f32, .bf16, .f16, .f64 => |tag| {
            const slice = buf.as_slice(tag.StorageType());
            for (slice, 0..) |*v, i| {
                v.* = tag.encode_f32(offset + scale * @as(f32, @floatFromInt(i % 1024)));
            }
        },
        else => @panic("fill_pattern: unsupported dtype"),
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
fn fill_attention_mask(buf: *zg.HostBuffer, batch: usize, seq: usize, active_len: usize) void {
    switch (buf.dtype) {
        inline .f32, .bf16, .f16, .f64 => |tag| {
            const T = tag.StorageType();
            const slice = buf.as_slice(T);
            @memset(slice, tag.encode_f32(0.0));
            const count = @min(seq, active_len);
            const one = tag.encode_f32(1.0);
            for (0..batch) |b| for (0..count) |i| {
                slice[b * seq + i] = one;
            };
        },
        else => @panic("fill_attention_mask: unsupported dtype"),
    }
}

/// Fill a [seq, seq] lower-triangular causal mask: 1.0 where col <= row, 0.0 above the diagonal.
/// Dtype-generic.
fn fill_causal_mask(buf: *zg.HostBuffer, seq: usize) void {
    switch (buf.dtype) {
        inline .f32, .bf16, .f16, .f64 => |tag| {
            const T = tag.StorageType();
            const slice = buf.as_slice(T);
            @memset(slice, tag.encode_f32(0.0));
            const one = tag.encode_f32(1.0);
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
/// dtype. Applies LLaMA 3 wavelength-based frequency correction via `llama3_rope_freq_correction`.
/// Base frequency is 500000.0 (LLaMA 3.2-1B config).
fn fill_rope_tables(sin: *zg.HostBuffer, cos: *zg.HostBuffer, seq: usize, head_dim: usize) void {
    const half = head_dim / 2;

    const base: f32 = 500000.0;
    var inv_freq: [32]f32 = undefined;
    for (0..half) |j| {
        const exp = @as(f32, @floatFromInt(2 * j)) / @as(f32, @floatFromInt(head_dim));
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
                s[i * half + j] = tag.encode_f32(@sin(theta));
                c[i * half + j] = tag.encode_f32(@cos(theta));
            };
        },
        else => @panic("fill_rope_tables: unsupported dtype"),
    }
}
