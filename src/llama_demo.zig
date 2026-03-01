const zg = @import("zigrad");
const stz = @import("safetensors_zg");
const llama_model = @import("llama_model.zig");
const ops = zg.pr.ops;
const std = @import("std");

const num_layers: usize = 16;

pub const LlamaKernelProvider = enum {
    mirage,
};

pub const LlamaDemoConfig = struct {
    train: bool,
    dtype: zg.pr.DType,
    seq: usize,
    batch: usize = 1,
    canonical_shapes: bool = false,
    execute_only: bool = false,
    kernel_provider: ?LlamaKernelProvider = null,
};

pub const LlamaDemoPipeline = enum {
    pr,
    mlir,
};

const upcast_loss = true; // bf16 logits over 128k vocab overflow bf16 range without this

fn loss_fn(params: anytype, batch: anytype) !zg.frontend.Tensor {
    return loss_fn_with_options(params, batch, .{});
}

fn loss_fn_mirage(params: anytype, batch: anytype) !zg.frontend.Tensor {
    return loss_fn_with_options(params, batch, .{ .kernelize_provider = "mirage" });
}

fn loss_fn_with_options(
    params: anytype,
    batch: anytype,
    forward_opts: llama_model.ForwardOptions,
) !zg.frontend.Tensor {
    const batch_size: usize = batch.x.tensor.shape.dims[0];
    const seq: usize = batch.x.tensor.shape.dims[1];
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
    const logits = try llama_model.forward_with_options(batch.x, batch.mask, batch.attention_mask, batch.sin, batch.cos, .{
        .w_emb = params.w_emb,
        .w_out = params.w_out,
        .norm = params.norm,
        .layers = layers[0..],
    }, 1e-6, forward_opts);
    const logits_f0 = if (upcast_loss and logits.tensor.dtype == .bf16) try logits.convert(.f32) else logits;
    const logits_f = logits_f0;
    const b = logits_f.builder;
    const attn_mask0 = if (batch.attention_mask.tensor.dtype == logits_f.tensor.dtype)
        batch.attention_mask
    else
        try batch.attention_mask.convert(logits_f.tensor.dtype);
    const attn_mask = attn_mask0;

    // Gather one vocab entry per (batch, seq) row without flattening.
    const row_b = try b.iota(.i32, &.{ batch_size, seq }, 0);
    const row_s = try b.iota(.i32, &.{ batch_size, seq }, 1);
    const tgt_ids = if (batch.target_ids.tensor.dtype == .i32)
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
    const max_b = try max_logits.broadcast_in_dim(logits_f.tensor.shape.dims, &.{ 0, 1 });
    const shifted = try logits_f.sub(max_b);

    // Match JAX-style logsumexp lowering: subtract in f32, exp in bf16 (when model dtype is bf16),
    // then accumulate reductions in f32.
    const shifted_f32 = if (shifted.tensor.dtype == .f32) shifted else try shifted.convert(.f32);
    const exp_in = if (logits.tensor.dtype == .bf16) try shifted_f32.convert(.bf16) else shifted_f32;
    const exp_logits = try exp_in.exp();
    const exp_logits_f32 = if (exp_logits.tensor.dtype == .f32) exp_logits else try exp_logits.convert(.f32);
    const sum_exp = try exp_logits_f32.reduce_sum(&.{2});
    const log_sum = try sum_exp.log();
    const max_f = if (max_logits.tensor.dtype == log_sum.tensor.dtype)
        max_logits
    else
        try max_logits.convert(log_sum.tensor.dtype);
    const logsumexp = try log_sum.add(max_f);

    const target_f = if (target_logits_2d.tensor.dtype == logsumexp.tensor.dtype)
        target_logits_2d
    else
        try target_logits_2d.convert(logsumexp.tensor.dtype);
    const loss_per = try logsumexp.sub(target_f);
    const loss_per_out = if (loss_per.tensor.dtype == logits_f.tensor.dtype)
        loss_per
    else
        try loss_per.convert(logits_f.tensor.dtype);

    const zero_lit = ops.types.scalar_literal(logits_f.tensor.dtype, 0.0);
    const zero = try logits_f.builder.scalar_literal(zero_lit);
    const zero_b = try zero.broadcast_in_dim(&.{ batch_size, seq }, &.{});
    const attn_zero = try attn_mask.compare(zero_b, .{ .direction = .GT, .compare_type = .FLOAT });
    const masked = try loss_per_out.select(attn_zero, zero_b);
    const masked_f = if (masked.tensor.dtype == .f32) masked else try masked.convert(.f32);
    const loss_sum = try masked_f.reduce_sum(&.{ 0, 1 });
    if (loss_sum.tensor.dtype == logits_f.tensor.dtype) return loss_sum;
    return loss_sum.convert(logits_f.tensor.dtype);
}

pub fn run_llama_ft_demo(
    allocator: std.mem.Allocator,
    plugin_path: []const u8,
    dump_pr: ?*zg.pipeline.DumpConfig,
    dump_mlir: ?*zg.pipeline.DumpConfig,
    warmup_steps: usize,
    steps: usize,
    quiet: bool,
    pipeline_kind: LlamaDemoPipeline,
    cfg: LlamaDemoConfig,
) !void {
    const TensorSpec = zg.frontend.TensorSpec;
    const train_mode = cfg.train;
    const model_dtype: zg.pr.DType = cfg.dtype;
    const host_dtype: zg.utils.DType = host_dtype_for(model_dtype);
    const batch_size: usize = cfg.batch;
    const execute_only = cfg.execute_only;

    const LayerSpec = struct {
        input_norm: TensorSpec,
        post_norm: TensorSpec,
        qkv_proj: TensorSpec,
        o_proj: TensorSpec,
        gate_proj: TensorSpec,
        up_proj: TensorSpec,
        down_proj: TensorSpec,
    };

    const ParamsSpec = struct {
        w_emb: TensorSpec,
        w_out: TensorSpec,
        norm: TensorSpec,
        layers: [num_layers]LayerSpec,
    };

    const BatchSpec = struct {
        x: TensorSpec,
        target_ids: TensorSpec,
        attention_mask: TensorSpec,
        mask: TensorSpec,
        sin: TensorSpec,
        cos: TensorSpec,
    };

    const seq: usize = cfg.seq;
    const vocab: usize = if (cfg.canonical_shapes) 4096 else 128256;
    const hidden: usize = if (cfg.canonical_shapes) 512 else 2048;
    const kv_out: usize = 512;
    const mlp_hidden: usize = hidden * 4;
    const qkv_out: usize = hidden + kv_out + kv_out;

    var layers_spec: [num_layers]LayerSpec = undefined;
    inline for (0..num_layers) |i| {
        layers_spec[i] = .{
            .input_norm = .{ .dtype = model_dtype, .dims = &.{hidden} },
            .post_norm = .{ .dtype = model_dtype, .dims = &.{hidden} },
            .qkv_proj = .{ .dtype = model_dtype, .dims = &.{ hidden, qkv_out } },
            .o_proj = .{ .dtype = model_dtype, .dims = &.{ hidden, hidden } },
            .gate_proj = .{ .dtype = model_dtype, .dims = &.{ hidden, mlp_hidden } },
            .up_proj = .{ .dtype = model_dtype, .dims = &.{ hidden, mlp_hidden } },
            .down_proj = .{ .dtype = model_dtype, .dims = &.{ mlp_hidden, hidden } },
        };
    }

    const params_spec = ParamsSpec{
        .w_emb = .{ .dtype = model_dtype, .dims = &.{ vocab, hidden } },
        .w_out = .{ .dtype = model_dtype, .dims = &.{ hidden, vocab } },
        .norm = .{ .dtype = model_dtype, .dims = &.{hidden} },
        .layers = layers_spec,
    };
    var dims_seq_seq: [2]usize = .{ seq, seq };
    var dims_seq_32: [2]usize = .{ seq, 32 };
    var dims_b_s: [2]usize = .{ batch_size, seq };
    const batch_spec = BatchSpec{
        .x = .{ .dtype = .i32, .dims = dims_b_s[0..] },
        .target_ids = .{ .dtype = .i32, .dims = dims_b_s[0..] },
        .attention_mask = .{ .dtype = model_dtype, .dims = dims_b_s[0..] },
        .mask = .{ .dtype = model_dtype, .dims = dims_seq_seq[0..] },
        .sin = .{ .dtype = model_dtype, .dims = dims_seq_32[0..] },
        .cos = .{ .dtype = model_dtype, .dims = dims_seq_32[0..] },
    };
    const inputs_spec = .{ params_spec, batch_spec };

    var compile_cfg = zg.frontend.CompileConfig{
        .entry_name = "llama_ft_step",
        .plugin_path = plugin_path,
        .dump_pr = if (dump_pr) |dump_cfg| dump_cfg.* else null,
        .dump_mlir = if (dump_mlir) |dump_cfg| dump_cfg.* else null,
    };

    const kernel_lane: zg.lower.KernelizationLane = switch (pipeline_kind) {
        .pr => .pr,
        .mlir => .mlir,
    };

    compile_cfg.lower.kernelization_lane = kernel_lane;
    if (compile_cfg.dump_mlir != null) {
        compile_cfg.lower.encoding = .text;
    }

    var kernel_registry: ?zg.kernel.KernelRegistry = null;
    defer if (kernel_registry) |*r| r.deinit();

    var kernel_package: ?zg.kernel.KernelPackage = null;
    defer if (kernel_package) |*p| p.deinit();

    var mirage_dispatch_state: ?zg.mirage.dispatch.MirageDispatchState = null;
    defer if (mirage_dispatch_state) |*s| s.deinit();

    var mirage_provider_impl: ?zg.mirage.provider.MirageProvider = null;
    var mirage_providers: [1]zg.kernel.KernelProvider = undefined;

    if (cfg.kernel_provider) |provider| {
        switch (provider) {
            .mirage => {
                kernel_registry = zg.kernel.KernelRegistry.init(allocator);
                kernel_package = zg.kernel.KernelPackage.init(allocator);
                mirage_dispatch_state = try zg.mirage.dispatch.MirageDispatchState.init(allocator);
                mirage_provider_impl = .{
                    .allocator = allocator,
                    .dispatch_state = &mirage_dispatch_state.?,
                };
                mirage_providers = .{mirage_provider_impl.?.kernel_provider()};

                compile_cfg.kernelize = .{
                    .registry = &kernel_registry.?,
                    .package = &kernel_package.?,
                    .providers = mirage_providers[0..],
                    .lane = kernel_lane,
                };
            },
        }
    }

    var backend_handle = try zg.frontend.init_backend(allocator, compile_cfg.plugin_path);
    defer backend_handle.deinit();

    const devices = try backend_handle.get_devices(allocator);
    defer allocator.free(devices);

    if (compile_cfg.device_index >= devices.len) return error.InvalidDeviceIndex;
    const device = &devices[compile_cfg.device_index];

    const param_count = 3 + num_layers * 7;

    const train = zg.frontend.train;
    const use_mirage_loss = cfg.kernel_provider != null;
    var compiled_train: ?train.CompiledTrainStep = null;
    var compiled_fwd: ?zg.frontend.CompiledForward = null;
    if (train_mode) {
        if (use_mirage_loss) {
            compiled_train = try train.compile_train_step(allocator, &backend_handle, device, loss_fn_mirage, inputs_spec, param_count, .{
                .optimizer = .{ .lr = 1e-4 },
                .compile = compile_cfg,
            });
        } else {
            compiled_train = try train.compile_train_step(allocator, &backend_handle, device, loss_fn, inputs_spec, param_count, .{
                .optimizer = .{ .lr = 1e-4 },
                .compile = compile_cfg,
            });
        }
    } else {
        if (use_mirage_loss) {
            compiled_fwd = try zg.frontend.compile_forward(allocator, &backend_handle, device, loss_fn_mirage, inputs_spec, compile_cfg);
        } else {
            compiled_fwd = try zg.frontend.compile_forward(allocator, &backend_handle, device, loss_fn, inputs_spec, compile_cfg);
        }
    }
    defer {
        if (compiled_train) |*ct| backend_handle.deinit_executable(&ct.exe);
        if (compiled_fwd) |*cf| backend_handle.deinit_executable(&cf.exe);
    }

    if (!quiet) {
        if (compiled_train) |*ct| log_compiled_memory_stats(&backend_handle, &ct.exe);
        if (compiled_fwd) |*cf| log_compiled_memory_stats(&backend_handle, &cf.exe);
    }

    const shape_w_emb = zg.utils.Shape{ .dims = &.{ vocab, hidden } };
    const shape_w_out = zg.utils.Shape{ .dims = &.{ hidden, vocab } };
    const shape_norm = zg.utils.Shape{ .dims = &.{hidden} };
    const shape_layer_norm = zg.utils.Shape{ .dims = &.{hidden} };
    const shape_q_proj = zg.utils.Shape{ .dims = &.{ hidden, hidden } };
    const shape_kv_proj = zg.utils.Shape{ .dims = &.{ hidden, kv_out } };
    const shape_qkv_proj = zg.utils.Shape{ .dims = &.{ hidden, qkv_out } };
    const shape_o_proj = zg.utils.Shape{ .dims = &.{ hidden, hidden } };
    const shape_gate_proj = zg.utils.Shape{ .dims = &.{ hidden, mlp_hidden } };
    const shape_up_proj = zg.utils.Shape{ .dims = &.{ hidden, mlp_hidden } };
    const shape_down_proj = zg.utils.Shape{ .dims = &.{ mlp_hidden, hidden } };
    const shape_x = zg.utils.Shape{ .dims = dims_b_s[0..] };
    const shape_target = zg.utils.Shape{ .dims = dims_b_s[0..] };
    const shape_mask = zg.utils.Shape{ .dims = dims_seq_seq[0..] };
    const shape_attn = zg.utils.Shape{ .dims = dims_b_s[0..] };
    const shape_rot = zg.utils.Shape{ .dims = dims_seq_32[0..] };

    var host_w_emb = try zg.utils.HostBuffer.init(allocator, shape_w_emb, host_dtype);
    defer host_w_emb.deinit();
    var host_w_out = try zg.utils.HostBuffer.init(allocator, shape_w_out, host_dtype);
    defer host_w_out.deinit();
    var host_norm = try zg.utils.HostBuffer.init(allocator, shape_norm, host_dtype);
    defer host_norm.deinit();
    var host_layer_input_norm: [num_layers]zg.utils.HostBuffer = undefined;
    var host_layer_post_norm: [num_layers]zg.utils.HostBuffer = undefined;
    var host_qkv_proj: [num_layers]zg.utils.HostBuffer = undefined;
    var host_o_proj: [num_layers]zg.utils.HostBuffer = undefined;
    var host_gate_proj: [num_layers]zg.utils.HostBuffer = undefined;
    var host_up_proj: [num_layers]zg.utils.HostBuffer = undefined;
    var host_down_proj: [num_layers]zg.utils.HostBuffer = undefined;

    var i: usize = 0;
    while (i < num_layers) : (i += 1) {
        host_layer_input_norm[i] = try zg.utils.HostBuffer.init(allocator, shape_layer_norm, host_dtype);
        host_layer_post_norm[i] = try zg.utils.HostBuffer.init(allocator, shape_layer_norm, host_dtype);
        host_qkv_proj[i] = try zg.utils.HostBuffer.init(allocator, shape_qkv_proj, host_dtype);
        host_o_proj[i] = try zg.utils.HostBuffer.init(allocator, shape_o_proj, host_dtype);
        host_gate_proj[i] = try zg.utils.HostBuffer.init(allocator, shape_gate_proj, host_dtype);
        host_up_proj[i] = try zg.utils.HostBuffer.init(allocator, shape_up_proj, host_dtype);
        host_down_proj[i] = try zg.utils.HostBuffer.init(allocator, shape_down_proj, host_dtype);
    }
    defer {
        var j: usize = 0;
        while (j < num_layers) : (j += 1) {
            host_layer_input_norm[j].deinit();
            host_layer_post_norm[j].deinit();
            host_qkv_proj[j].deinit();
            host_o_proj[j].deinit();
            host_gate_proj[j].deinit();
            host_up_proj[j].deinit();
            host_down_proj[j].deinit();
        }
    }
    var host_x = try zg.utils.HostBuffer.init(allocator, shape_x, .i32);
    defer host_x.deinit();
    var host_target_ids = try zg.utils.HostBuffer.init(allocator, shape_target, .i32);
    defer host_target_ids.deinit();
    var host_attention_mask = try zg.utils.HostBuffer.init(allocator, shape_attn, host_dtype);
    defer host_attention_mask.deinit();
    var host_mask = try zg.utils.HostBuffer.init(allocator, shape_mask, host_dtype);
    defer host_mask.deinit();
    var host_sin = try zg.utils.HostBuffer.init(allocator, shape_rot, host_dtype);
    defer host_sin.deinit();
    var host_cos = try zg.utils.HostBuffer.init(allocator, shape_rot, host_dtype);
    defer host_cos.deinit();

    const default_path = "./weights/llama-3.2-1b-instruct/model.safetensors";
    const weights_path = std.process.getEnvVarOwned(allocator, "ZG_LLAMA_SAFETENSORS_PATH") catch default_path;
    defer if (!std.mem.eql(u8, weights_path, default_path)) allocator.free(weights_path);

    const loaded_weights = load_llama_weights(
        allocator,
        weights_path,
        &host_w_emb,
        &host_w_out,
        &host_norm,
        host_layer_input_norm[0..],
        host_layer_post_norm[0..],
        host_qkv_proj[0..],
        host_o_proj[0..],
        host_gate_proj[0..],
        host_up_proj[0..],
        host_down_proj[0..],
        shape_w_emb.dims,
        shape_w_out.dims,
        shape_norm.dims,
        shape_layer_norm.dims,
        shape_qkv_proj.dims,
        shape_q_proj.dims,
        shape_kv_proj.dims,
        shape_o_proj.dims,
        shape_gate_proj.dims,
        shape_up_proj.dims,
        shape_down_proj.dims,
    ) catch |err| switch (err) {
        error.TensorShapeMismatch,
        error.TensorSizeMismatch,
        error.TensorDtypeMismatch,
        stz.Error.TensorNotFound,
        => false,
        else => return err,
    };

    if (loaded_weights) {
        if (!quiet) {
            std.log.info("llama-ft-demo: loaded weights from {s}", .{weights_path});
        }
    } else {
        if (host_dtype == .bf16) {
            fill_pattern_bf16(host_w_emb.as_slice(u16), 1e-3, 0.0);
            fill_pattern_bf16(host_w_out.as_slice(u16), 1e-3, 0.0);
            fill_pattern_bf16(host_norm.as_slice(u16), 1e-3, 0.0);
            var k: usize = 0;
            while (k < num_layers) : (k += 1) {
                fill_pattern_bf16(host_layer_input_norm[k].as_slice(u16), 1e-3, 0.0);
                fill_pattern_bf16(host_layer_post_norm[k].as_slice(u16), 1e-3, 0.0);
                fill_pattern_bf16(host_qkv_proj[k].as_slice(u16), 1e-3, 0.0);
                fill_pattern_bf16(host_o_proj[k].as_slice(u16), 1e-3, 0.0);
                fill_pattern_bf16(host_gate_proj[k].as_slice(u16), 1e-3, 0.0);
                fill_pattern_bf16(host_up_proj[k].as_slice(u16), 1e-3, 0.0);
                fill_pattern_bf16(host_down_proj[k].as_slice(u16), 1e-3, 0.0);
            }
        } else {
            fill_pattern(host_w_emb.as_slice(f32), 1e-3, 0.0);
            fill_pattern(host_w_out.as_slice(f32), 1e-3, 0.0);
            fill_pattern(host_norm.as_slice(f32), 1e-3, 0.0);
            var k: usize = 0;
            while (k < num_layers) : (k += 1) {
                fill_pattern(host_layer_input_norm[k].as_slice(f32), 1e-3, 0.0);
                fill_pattern(host_layer_post_norm[k].as_slice(f32), 1e-3, 0.0);
                fill_pattern(host_qkv_proj[k].as_slice(f32), 1e-3, 0.0);
                fill_pattern(host_o_proj[k].as_slice(f32), 1e-3, 0.0);
                fill_pattern(host_gate_proj[k].as_slice(f32), 1e-3, 0.0);
                fill_pattern(host_up_proj[k].as_slice(f32), 1e-3, 0.0);
                fill_pattern(host_down_proj[k].as_slice(f32), 1e-3, 0.0);
            }
        }
        std.log.warn("llama-ft-demo: using synthetic weights (set ZG_LLAMA_SAFETENSORS_PATH)", .{});
    }

    const token_seed = [_]usize{ 128000, 128009, 128001, 128008 };
    const target_seed = [_]usize{ 128009, 128001, 128008, 128001 };
    var tokens: [token_seed.len]usize = undefined;
    var targets: [target_seed.len]usize = undefined;
    for (token_seed, 0..) |value, idx| {
        tokens[idx] = value % vocab;
    }
    for (target_seed, 0..) |value, idx| {
        targets[idx] = value % vocab;
    }
    fill_i32_tokens_batched(host_x.as_slice(i32), batch_size, seq, &tokens);
    fill_i32_tokens_batched(host_target_ids.as_slice(i32), batch_size, seq, &targets);
    if (host_dtype == .bf16) {
        fill_attention_mask_bf16_batched(host_attention_mask.as_slice(u16), batch_size, seq, tokens.len);
        fill_causal_mask_bf16(host_mask.as_slice(u16), seq);
    } else {
        fill_attention_mask_batched(host_attention_mask.as_slice(f32), batch_size, seq, tokens.len);
        fill_causal_mask(host_mask.as_slice(f32), seq);
    }
    if (host_dtype == .bf16) {
        fill_rope_tables_bf16(host_sin.as_slice(u16), host_cos.as_slice(u16), seq, 64);
    } else {
        fill_rope_tables(host_sin.as_slice(f32), host_cos.as_slice(f32), seq, 64);
    }

    var total_ns: u64 = 0;

    const upload = zg.frontend.upload_host_buffer;
    const tmp_w_emb = try upload(allocator, &backend_handle, device, &host_w_emb);
    const tmp_w_out = try upload(allocator, &backend_handle, device, &host_w_out);
    const tmp_norm = try upload(allocator, &backend_handle, device, &host_norm);
    var tmp_layer_input_norm: [num_layers]zg.backend.pjrt.Buffer = undefined;
    var tmp_layer_post_norm: [num_layers]zg.backend.pjrt.Buffer = undefined;
    var tmp_qkv_proj: [num_layers]zg.backend.pjrt.Buffer = undefined;
    var tmp_o_proj: [num_layers]zg.backend.pjrt.Buffer = undefined;
    var tmp_gate_proj: [num_layers]zg.backend.pjrt.Buffer = undefined;
    var tmp_up_proj: [num_layers]zg.backend.pjrt.Buffer = undefined;
    var tmp_down_proj: [num_layers]zg.backend.pjrt.Buffer = undefined;

    var m: usize = 0;
    while (m < num_layers) : (m += 1) {
        tmp_layer_input_norm[m] = try upload(allocator, &backend_handle, device, &host_layer_input_norm[m]);
        tmp_layer_post_norm[m] = try upload(allocator, &backend_handle, device, &host_layer_post_norm[m]);
        tmp_qkv_proj[m] = try upload(allocator, &backend_handle, device, &host_qkv_proj[m]);
        tmp_o_proj[m] = try upload(allocator, &backend_handle, device, &host_o_proj[m]);
        tmp_gate_proj[m] = try upload(allocator, &backend_handle, device, &host_gate_proj[m]);
        tmp_up_proj[m] = try upload(allocator, &backend_handle, device, &host_up_proj[m]);
        tmp_down_proj[m] = try upload(allocator, &backend_handle, device, &host_down_proj[m]);
    }
    const tmp_x = try upload(allocator, &backend_handle, device, &host_x);
    const tmp_target_ids = try upload(allocator, &backend_handle, device, &host_target_ids);
    const tmp_attention_mask = try upload(allocator, &backend_handle, device, &host_attention_mask);
    const tmp_mask = try upload(allocator, &backend_handle, device, &host_mask);
    const tmp_sin = try upload(allocator, &backend_handle, device, &host_sin);
    const tmp_cos = try upload(allocator, &backend_handle, device, &host_cos);

    const loss_dtype: zg.utils.DType = if (upcast_loss) .f32 else host_dtype;
    var loss_host = try zg.utils.HostBuffer.init(allocator, .{ .dims = &.{} }, loss_dtype);
    defer loss_host.deinit();

    // Build param and batch buffer arrays.
    var param_bufs = std.ArrayList(zg.backend.pjrt.RawBuffer).empty;
    defer param_bufs.deinit(allocator);
    try param_bufs.append(allocator, tmp_w_emb.pjrt_buffer);
    try param_bufs.append(allocator, tmp_w_out.pjrt_buffer);
    try param_bufs.append(allocator, tmp_norm.pjrt_buffer);
    var p: usize = 0;
    while (p < num_layers) : (p += 1) {
        try param_bufs.append(allocator, tmp_layer_input_norm[p].pjrt_buffer);
        try param_bufs.append(allocator, tmp_layer_post_norm[p].pjrt_buffer);
        try param_bufs.append(allocator, tmp_qkv_proj[p].pjrt_buffer);
        try param_bufs.append(allocator, tmp_o_proj[p].pjrt_buffer);
        try param_bufs.append(allocator, tmp_gate_proj[p].pjrt_buffer);
        try param_bufs.append(allocator, tmp_up_proj[p].pjrt_buffer);
        try param_bufs.append(allocator, tmp_down_proj[p].pjrt_buffer);
    }

    const batch_bufs = [_]zg.backend.pjrt.RawBuffer{
        tmp_x.pjrt_buffer,
        tmp_target_ids.pjrt_buffer,
        tmp_attention_mask.pjrt_buffer,
        tmp_mask.pjrt_buffer,
        tmp_sin.pjrt_buffer,
        tmp_cos.pjrt_buffer,
    };

    const is_cpu = try backend_handle.buffer_is_on_cpu(&(zg.backend.pjrt.Buffer{ .pjrt_buffer = tmp_w_emb.pjrt_buffer }));

    if (train_mode) {
        // Use TrainState for the training path.
        var state = try train.TrainState.init(
            allocator,
            &compiled_train.?,
            &backend_handle,
            param_bufs.items,
            &batch_bufs,
        );
        defer state.deinit();
        // Batch buffers are not owned by TrainState.
        defer for (batch_bufs) |raw| {
            var buf = zg.backend.pjrt.Buffer{ .pjrt_buffer = raw };
            backend_handle.deinit_buffer(&buf);
        };
        // param_bufs ownership transferred to TrainState; clear to avoid double-free.
        param_bufs.clearRetainingCapacity();

        var warmup: usize = 0;
        while (warmup < warmup_steps) : (warmup += 1) {
            var result = try state.step();
            if (result.event) |e| {
                var ev = e;
                try backend_handle.await_event(&ev);
                backend_handle.deinit_event(&ev);
            }
            if (!execute_only and !is_cpu and !quiet) {
                var loss_ev = try backend_handle.buffer_to_host(&result.loss_buf, loss_host.data);
                try backend_handle.await_event(&loss_ev);
                backend_handle.deinit_event(&loss_ev);
            }
            backend_handle.deinit_buffer(&result.loss_buf);
        }

        const nvtx_label: [:0]const u8 = "llama-ft-demo timed loop";
        var nvtx_range = NvtxRange.init() catch null;
        defer if (nvtx_range) |*range| range.deinit();
        if (nvtx_range) |*range| range.push(nvtx_label) catch {};

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
            const exec_ns = timer.lap();

            const loss: ?f32 = if (quiet or execute_only) null else if (is_cpu) blk: {
                if (loss_dtype == .bf16) {
                    const ptr: [*]const u16 = @ptrFromInt(try backend_handle.buffer_unsafe_pointer(&result.loss_buf));
                    break :blk bf16_to_f32(ptr[0]);
                }
                const ptr: [*]const f32 = @ptrFromInt(try backend_handle.buffer_unsafe_pointer(&result.loss_buf));
                break :blk ptr[0];
            } else blk: {
                var loss_ev = try backend_handle.buffer_to_host(&result.loss_buf, loss_host.data);
                try backend_handle.await_event(&loss_ev);
                backend_handle.deinit_event(&loss_ev);
                break :blk if (loss_dtype == .bf16)
                    bf16_to_f32(loss_host.as_slice(u16)[0])
                else
                    loss_host.as_slice(f32)[0];
            };
            const loss_read_ns = timer.lap();

            backend_handle.deinit_buffer(&result.loss_buf);

            const cleanup_ns = timer.lap();
            const step_ns = dispatch_ns + exec_ns + loss_read_ns + cleanup_ns;
            total_ns += step_ns;
            const step_ms = @as(f64, @floatFromInt(step_ns)) / std.time.ns_per_ms;
            const dispatch_ms = @as(f64, @floatFromInt(dispatch_ns)) / std.time.ns_per_ms;
            const exec_ms = @as(f64, @floatFromInt(exec_ns)) / std.time.ns_per_ms;
            const loss_ms = @as(f64, @floatFromInt(loss_read_ns)) / std.time.ns_per_ms;
            const cleanup_ms = @as(f64, @floatFromInt(cleanup_ns)) / std.time.ns_per_ms;
            if (!quiet) {
                if (loss) |loss_value| {
                    std.log.info("llama-ft-demo step {d}: loss={d:.6} dispatch={d:.3}ms exec={d:.3}ms host_read={d:.3}ms cleanup={d:.3}ms total={d:.3}ms", .{
                        step, loss_value, dispatch_ms, exec_ms, loss_ms, cleanup_ms, step_ms,
                    });
                } else {
                    std.log.info("llama-ft-demo step {d}: dispatch={d:.3}ms exec={d:.3}ms host_read={d:.3}ms cleanup={d:.3}ms total={d:.3}ms", .{
                        step, dispatch_ms, exec_ms, loss_ms, cleanup_ms, step_ms,
                    });
                }
            }
        }

        if (nvtx_range) |*range| range.pop() catch {};
    } else {
        // Forward-only mode: simple execute loop, no parameter swapping.
        var fwd_exe = compiled_fwd.?.exe;

        var input_ptrs = std.ArrayList(zg.backend.pjrt.RawBuffer).empty;
        defer input_ptrs.deinit(allocator);
        try input_ptrs.appendSlice(allocator, param_bufs.items);
        for (batch_bufs) |b| try input_ptrs.append(allocator, b);

        defer {
            for (input_ptrs.items) |raw| {
                var buf = zg.backend.pjrt.Buffer{ .pjrt_buffer = raw };
                backend_handle.deinit_buffer(&buf);
            }
        }
        // param_bufs ownership transferred to input_ptrs; clear to avoid double-free.
        param_bufs.clearRetainingCapacity();

        var output_ptrs = [1]?zg.backend.pjrt.RawBuffer{null};

        var warmup: usize = 0;
        while (warmup < warmup_steps) : (warmup += 1) {
            @memset(output_ptrs[0..], null);
            const ev = try backend_handle.execute_into(&fwd_exe, input_ptrs.items, &output_ptrs, null);
            const loss_raw = output_ptrs[0] orelse return error.PjrtReturnedNullOutputBuffer;
            var loss_buf = zg.backend.pjrt.Buffer{ .pjrt_buffer = loss_raw };
            if (ev) |e| {
                var evv = e;
                try backend_handle.await_event(&evv);
                backend_handle.deinit_event(&evv);
            }
            if (!execute_only and !is_cpu and !quiet) {
                var loss_ev = try backend_handle.buffer_to_host(&loss_buf, loss_host.data);
                try backend_handle.await_event(&loss_ev);
                backend_handle.deinit_event(&loss_ev);
            }
            backend_handle.deinit_buffer(&loss_buf);
        }

        const nvtx_label: [:0]const u8 = "llama-ft-demo timed loop";
        var nvtx_range = NvtxRange.init() catch null;
        defer if (nvtx_range) |*range| range.deinit();
        if (nvtx_range) |*range| range.push(nvtx_label) catch {};

        var step: usize = 0;
        while (step < steps) : (step += 1) {
            var timer = try std.time.Timer.start();
            @memset(output_ptrs[0..], null);
            const event = try backend_handle.execute_into(&fwd_exe, input_ptrs.items, &output_ptrs, null);
            const dispatch_ns = timer.lap();

            if (event) |ev| {
                var evv = ev;
                try backend_handle.await_event(&evv);
                backend_handle.deinit_event(&evv);
            }
            const exec_ns = timer.lap();

            const loss_raw2 = output_ptrs[0] orelse return error.PjrtReturnedNullOutputBuffer;
            var loss_buf = zg.backend.pjrt.Buffer{ .pjrt_buffer = loss_raw2 };
            const loss: ?f32 = if (quiet or execute_only) null else if (is_cpu) blk: {
                if (loss_dtype == .bf16) {
                    const ptr: [*]const u16 = @ptrFromInt(try backend_handle.buffer_unsafe_pointer(&loss_buf));
                    break :blk bf16_to_f32(ptr[0]);
                }
                const ptr: [*]const f32 = @ptrFromInt(try backend_handle.buffer_unsafe_pointer(&loss_buf));
                break :blk ptr[0];
            } else blk: {
                var loss_ev = try backend_handle.buffer_to_host(&loss_buf, loss_host.data);
                try backend_handle.await_event(&loss_ev);
                backend_handle.deinit_event(&loss_ev);
                break :blk if (loss_dtype == .bf16)
                    bf16_to_f32(loss_host.as_slice(u16)[0])
                else
                    loss_host.as_slice(f32)[0];
            };
            const loss_read_ns = timer.lap();

            backend_handle.deinit_buffer(&loss_buf);

            const cleanup_ns = timer.lap();
            const step_ns = dispatch_ns + exec_ns + loss_read_ns + cleanup_ns;
            total_ns += step_ns;
            const step_ms = @as(f64, @floatFromInt(step_ns)) / std.time.ns_per_ms;
            const dispatch_ms = @as(f64, @floatFromInt(dispatch_ns)) / std.time.ns_per_ms;
            const exec_ms = @as(f64, @floatFromInt(exec_ns)) / std.time.ns_per_ms;
            const loss_ms = @as(f64, @floatFromInt(loss_read_ns)) / std.time.ns_per_ms;
            const cleanup_ms = @as(f64, @floatFromInt(cleanup_ns)) / std.time.ns_per_ms;
            if (!quiet) {
                if (loss) |loss_value| {
                    std.log.info("llama-ft-demo step {d}: loss={d:.6} dispatch={d:.3}ms exec={d:.3}ms host_read={d:.3}ms cleanup={d:.3}ms total={d:.3}ms", .{
                        step, loss_value, dispatch_ms, exec_ms, loss_ms, cleanup_ms, step_ms,
                    });
                } else {
                    std.log.info("llama-ft-demo step {d}: dispatch={d:.3}ms exec={d:.3}ms host_read={d:.3}ms cleanup={d:.3}ms total={d:.3}ms", .{
                        step, dispatch_ms, exec_ms, loss_ms, cleanup_ms, step_ms,
                    });
                }
            }
        }

        if (nvtx_range) |*range| range.pop() catch {};
    }

    const avg_ms = @as(f64, @floatFromInt(total_ns)) / std.time.ns_per_ms / @as(f64, @floatFromInt(steps));
    if (!quiet) {
        log_device_memory_stats(&backend_handle, device);
    }
    std.log.info("llama-ft-demo avg_step_ms={d:.3} (warmup={d} steps={d} seq={d})", .{ avg_ms, warmup_steps, steps, seq });
    std.log.info("OK: llama-ft-demo executed", .{});
}

const NvtxRange = struct {
    lib: std.DynLib,
    push_fn: *const fn ([*:0]const u8) callconv(.c) c_int,
    pop_fn: *const fn () callconv(.c) c_int,

    pub fn init() !NvtxRange {
        if (std.process.getEnvVarOwned(std.heap.page_allocator, "ZG_EXTERNAL_SDK_ROOT")) |sdk| {
            defer std.heap.page_allocator.free(sdk);
            var buf1: [1024]u8 = undefined;
            var buf2: [1024]u8 = undefined;
            const p1 = std.fmt.bufPrintZ(&buf1, "{s}/runtime/nvidia/nvtx/lib/libnvToolsExt.so", .{sdk}) catch null;
            if (p1) |p| {
                if (open_nvtx(p)) |range| return range;
            }
            const p2 = std.fmt.bufPrintZ(&buf2, "{s}/runtime/nvidia/nvtx/lib/libnvToolsExt.so.1", .{sdk}) catch null;
            if (p2) |p| {
                if (open_nvtx(p)) |range| return range;
            }
        } else |_| {}

        const names = [_][]const u8{
            "libnvToolsExt.so",
            "libnvToolsExt.so.1",
            "libnvToolsExt.so.1.0",
        };
        var i: usize = 0;
        while (i < names.len) : (i += 1) {
            if (open_nvtx(names[i])) |range| return range;
        }
        return error.FileNotFound;
    }

    pub fn deinit(self: *NvtxRange) void {
        self.lib.close();
    }

    pub fn push(self: *NvtxRange, label: [:0]const u8) !void {
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

fn load_llama_weights(
    allocator: std.mem.Allocator,
    path: []const u8,
    w_emb: *zg.utils.HostBuffer,
    w_out: *zg.utils.HostBuffer,
    norm: *zg.utils.HostBuffer,
    layer_input_norm: []zg.utils.HostBuffer,
    layer_post_norm: []zg.utils.HostBuffer,
    qkv_proj: []zg.utils.HostBuffer,
    o_proj: []zg.utils.HostBuffer,
    gate_proj: []zg.utils.HostBuffer,
    up_proj: []zg.utils.HostBuffer,
    down_proj: []zg.utils.HostBuffer,
    shape_w_emb: []const usize,
    shape_w_out: []const usize,
    shape_norm: []const usize,
    shape_layer_norm: []const usize,
    shape_qkv_proj: []const usize,
    shape_q_proj: []const usize,
    shape_k_proj: []const usize,
    shape_o_proj: []const usize,
    shape_gate_proj: []const usize,
    shape_up_proj: []const usize,
    shape_down_proj: []const usize,
) !bool {
    const data = read_file_aligned(allocator, path) catch return false;
    defer allocator.free(data);

    var st_file = try stz.SafeTensorsFile.deserialize(data, allocator);
    defer st_file.deinit();

    const emb_view = try st_file.get("model.embed_tokens.weight");
    if (w_emb.dtype == .bf16) {
        try copy_tensor_to_bf16(emb_view, w_emb.as_slice(u16), shape_w_emb);
    } else {
        try copy_tensor_to_f32(emb_view, w_emb.as_slice(f32), shape_w_emb);
    }

    const norm_view = try st_file.get("model.norm.weight");
    if (norm.dtype == .bf16) {
        try copy_tensor_to_bf16(norm_view, norm.as_slice(u16), shape_norm);
    } else {
        try copy_tensor_to_f32(norm_view, norm.as_slice(f32), shape_norm);
    }

    if (layer_input_norm.len != num_layers) return error.InvalidParams;
    if (layer_post_norm.len != num_layers) return error.InvalidParams;

    inline for (0..num_layers) |i| {
        const prefix = try std.fmt.allocPrint(allocator, "model.layers.{d}.", .{i});
        defer allocator.free(prefix);

        const in_name = try std.fmt.allocPrint(allocator, "{s}input_layernorm.weight", .{prefix});
        defer allocator.free(in_name);
        const post_name = try std.fmt.allocPrint(allocator, "{s}post_attention_layernorm.weight", .{prefix});
        defer allocator.free(post_name);
        const q_name = try std.fmt.allocPrint(allocator, "{s}self_attn.q_proj.weight", .{prefix});
        defer allocator.free(q_name);
        const k_name = try std.fmt.allocPrint(allocator, "{s}self_attn.k_proj.weight", .{prefix});
        defer allocator.free(k_name);
        const v_name = try std.fmt.allocPrint(allocator, "{s}self_attn.v_proj.weight", .{prefix});
        defer allocator.free(v_name);
        const o_name = try std.fmt.allocPrint(allocator, "{s}self_attn.o_proj.weight", .{prefix});
        defer allocator.free(o_name);
        const gate_name = try std.fmt.allocPrint(allocator, "{s}mlp.gate_proj.weight", .{prefix});
        defer allocator.free(gate_name);
        const up_name = try std.fmt.allocPrint(allocator, "{s}mlp.up_proj.weight", .{prefix});
        defer allocator.free(up_name);
        const down_name = try std.fmt.allocPrint(allocator, "{s}mlp.down_proj.weight", .{prefix});
        defer allocator.free(down_name);

        const layer_in = try st_file.get(in_name);
        if (layer_input_norm[i].dtype == .bf16) {
            try copy_tensor_to_bf16(layer_in, layer_input_norm[i].as_slice(u16), shape_layer_norm);
        } else {
            try copy_tensor_to_f32(layer_in, layer_input_norm[i].as_slice(f32), shape_layer_norm);
        }
        const layer_post = try st_file.get(post_name);
        if (layer_post_norm[i].dtype == .bf16) {
            try copy_tensor_to_bf16(layer_post, layer_post_norm[i].as_slice(u16), shape_layer_norm);
        } else {
            try copy_tensor_to_f32(layer_post, layer_post_norm[i].as_slice(f32), shape_layer_norm);
        }

        const q_view = try st_file.get(q_name);
        const k_view = try st_file.get(k_name);
        const v_view = try st_file.get(v_name);
        const q_cols = shape_q_proj[1];
        const k_cols = shape_k_proj[1];
        if (qkv_proj[i].dtype == .bf16) {
            const dst = qkv_proj[i].as_slice(u16);
            try copy_tensor_to_bf16_transposed_into_cols(q_view, dst, shape_qkv_proj, 0, shape_q_proj);
            try copy_tensor_to_bf16_transposed_into_cols(k_view, dst, shape_qkv_proj, q_cols, shape_k_proj);
            try copy_tensor_to_bf16_transposed_into_cols(v_view, dst, shape_qkv_proj, q_cols + k_cols, shape_k_proj);
        } else {
            const dst = qkv_proj[i].as_slice(f32);
            try copy_tensor_to_f32_transposed_into_cols(q_view, dst, shape_qkv_proj, 0, shape_q_proj);
            try copy_tensor_to_f32_transposed_into_cols(k_view, dst, shape_qkv_proj, q_cols, shape_k_proj);
            try copy_tensor_to_f32_transposed_into_cols(v_view, dst, shape_qkv_proj, q_cols + k_cols, shape_k_proj);
        }
        const o_view = try st_file.get(o_name);
        if (o_proj[i].dtype == .bf16) {
            try copy_tensor_to_bf16_transposed(o_view, o_proj[i].as_slice(u16), shape_o_proj);
        } else {
            try copy_tensor_to_f32_transposed(o_view, o_proj[i].as_slice(f32), shape_o_proj);
        }

        const gate_view = try st_file.get(gate_name);
        if (gate_proj[i].dtype == .bf16) {
            try copy_tensor_to_bf16_transposed(gate_view, gate_proj[i].as_slice(u16), shape_gate_proj);
        } else {
            try copy_tensor_to_f32_transposed(gate_view, gate_proj[i].as_slice(f32), shape_gate_proj);
        }
        const up_view = try st_file.get(up_name);
        if (up_proj[i].dtype == .bf16) {
            try copy_tensor_to_bf16_transposed(up_view, up_proj[i].as_slice(u16), shape_up_proj);
        } else {
            try copy_tensor_to_f32_transposed(up_view, up_proj[i].as_slice(f32), shape_up_proj);
        }
        const down_view = try st_file.get(down_name);
        if (down_proj[i].dtype == .bf16) {
            try copy_tensor_to_bf16_transposed(down_view, down_proj[i].as_slice(u16), shape_down_proj);
        } else {
            try copy_tensor_to_f32_transposed(down_view, down_proj[i].as_slice(f32), shape_down_proj);
        }
    }

    const lm_view = get_optional(&st_file, "lm_head.weight") catch |err| return err;
    if (lm_view) |view| {
        if (w_out.dtype == .bf16) {
            try copy_tensor_to_bf16_transposed(view, w_out.as_slice(u16), shape_w_out);
        } else {
            try copy_tensor_to_f32_transposed(view, w_out.as_slice(f32), shape_w_out);
        }
    } else {
        if (w_out.dtype == .bf16) {
            try transpose_vocab_hidden_bf16(w_emb.as_slice(u16), w_out.as_slice(u16), shape_w_emb, shape_w_out);
        } else {
            try transpose_vocab_hidden(w_emb.as_slice(f32), w_out.as_slice(f32), shape_w_emb, shape_w_out);
        }
    }

    return true;
}

fn get_optional(file: *stz.SafeTensorsFile, name: []const u8) !?stz.TensorView {
    const view = file.get(name) catch |err| switch (err) {
        stz.Error.TensorNotFound => return null,
        else => return err,
    };
    return view;
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

fn copy_tensor_to_f32(view: stz.TensorView, out: []f32, expected_shape: []const usize) !void {
    if (!std.mem.eql(usize, view.info.shape, expected_shape)) return error.TensorShapeMismatch;

    var count: usize = 1;
    for (view.info.shape) |d| count *= d;
    if (count != out.len) return error.TensorSizeMismatch;

    switch (view.info.dtype) {
        .f32 => {
            const data = std.mem.bytesAsSlice(f32, view.data);
            if (data.len != out.len) return error.TensorSizeMismatch;
            @memcpy(out, data);
        },
        .bf16 => {
            const data = std.mem.bytesAsSlice(u16, view.data);
            if (data.len != out.len) return error.TensorSizeMismatch;
            for (data, 0..) |v, i| {
                out[i] = bf16_to_f32(v);
            }
        },
        else => return error.TensorDtypeMismatch,
    }
}

fn copy_tensor_to_bf16(view: stz.TensorView, out: []u16, expected_shape: []const usize) !void {
    if (!std.mem.eql(usize, view.info.shape, expected_shape)) return error.TensorShapeMismatch;

    var count: usize = 1;
    for (view.info.shape) |d| count *= d;
    if (count != out.len) return error.TensorSizeMismatch;

    switch (view.info.dtype) {
        .bf16 => {
            const data = std.mem.bytesAsSlice(u16, view.data);
            if (data.len != out.len) return error.TensorSizeMismatch;
            @memcpy(out, data);
        },
        .f32 => {
            const data = std.mem.bytesAsSlice(f32, view.data);
            if (data.len != out.len) return error.TensorSizeMismatch;
            for (data, 0..) |v, i| {
                out[i] = f32_to_bf16(v);
            }
        },
        else => return error.TensorDtypeMismatch,
    }
}

fn copy_tensor_to_f32_transposed(view: stz.TensorView, out: []f32, expected_shape: []const usize) !void {
    if (view.info.shape.len != 2 or expected_shape.len != 2) return error.TensorShapeMismatch;
    if (view.info.shape[0] != expected_shape[1] or view.info.shape[1] != expected_shape[0]) {
        return error.TensorShapeMismatch;
    }

    const tmp = try std.heap.raw_c_allocator.alloc(f32, view.info.shape[0] * view.info.shape[1]);
    defer std.heap.raw_c_allocator.free(tmp);
    try copy_tensor_to_f32(view, tmp, view.info.shape);
    try transpose_vocab_hidden(tmp, out, view.info.shape, expected_shape);
}

fn copy_tensor_to_bf16_transposed(view: stz.TensorView, out: []u16, expected_shape: []const usize) !void {
    if (view.info.shape.len != 2 or expected_shape.len != 2) return error.TensorShapeMismatch;
    if (view.info.shape[0] != expected_shape[1] or view.info.shape[1] != expected_shape[0]) {
        return error.TensorShapeMismatch;
    }

    switch (view.info.dtype) {
        .bf16 => {
            const data = std.mem.bytesAsSlice(u16, view.data);
            if (data.len != view.info.shape[0] * view.info.shape[1]) return error.TensorSizeMismatch;
            try transpose_vocab_hidden_bf16(data, out, view.info.shape, expected_shape);
        },
        .f32 => {
            const data = std.mem.bytesAsSlice(f32, view.data);
            if (data.len != view.info.shape[0] * view.info.shape[1]) return error.TensorSizeMismatch;

            const vocab = view.info.shape[0];
            const hidden = view.info.shape[1];
            if (expected_shape[0] != hidden or expected_shape[1] != vocab) return error.TensorShapeMismatch;

            var v: usize = 0;
            while (v < vocab) : (v += 1) {
                var h: usize = 0;
                while (h < hidden) : (h += 1) {
                    out[h * vocab + v] = f32_to_bf16(data[v * hidden + h]);
                }
            }
        },
        else => return error.TensorDtypeMismatch,
    }
}

fn copy_tensor_to_bf16_transposed_into_cols(
    view: stz.TensorView,
    dst: []u16,
    dst_shape: []const usize,
    col_offset: usize,
    expected_shape: []const usize,
) !void {
    if (view.info.shape.len != 2 or expected_shape.len != 2 or dst_shape.len != 2) return error.TensorShapeMismatch;
    if (view.info.shape[0] != expected_shape[1] or view.info.shape[1] != expected_shape[0]) {
        return error.TensorShapeMismatch;
    }
    const in_dim = expected_shape[0];
    const out_dim = expected_shape[1];
    if (dst_shape[0] != in_dim) return error.TensorShapeMismatch;
    if (col_offset + out_dim > dst_shape[1]) return error.TensorShapeMismatch;
    if (dst.len != dst_shape[0] * dst_shape[1]) return error.TensorSizeMismatch;

    const dst_stride = dst_shape[1];
    switch (view.info.dtype) {
        .bf16 => {
            const src = std.mem.bytesAsSlice(u16, view.data);
            if (src.len != view.info.shape[0] * view.info.shape[1]) return error.TensorSizeMismatch;

            var o: usize = 0;
            while (o < out_dim) : (o += 1) {
                var i: usize = 0;
                while (i < in_dim) : (i += 1) {
                    dst[i * dst_stride + (col_offset + o)] = src[o * in_dim + i];
                }
            }
        },
        .f32 => {
            const src = std.mem.bytesAsSlice(f32, view.data);
            if (src.len != view.info.shape[0] * view.info.shape[1]) return error.TensorSizeMismatch;

            var o: usize = 0;
            while (o < out_dim) : (o += 1) {
                var i: usize = 0;
                while (i < in_dim) : (i += 1) {
                    dst[i * dst_stride + (col_offset + o)] = f32_to_bf16(src[o * in_dim + i]);
                }
            }
        },
        else => return error.TensorDtypeMismatch,
    }
}

fn copy_tensor_to_f32_transposed_into_cols(
    view: stz.TensorView,
    dst: []f32,
    dst_shape: []const usize,
    col_offset: usize,
    expected_shape: []const usize,
) !void {
    if (view.info.shape.len != 2 or expected_shape.len != 2 or dst_shape.len != 2) return error.TensorShapeMismatch;
    if (view.info.shape[0] != expected_shape[1] or view.info.shape[1] != expected_shape[0]) {
        return error.TensorShapeMismatch;
    }
    const in_dim = expected_shape[0];
    const out_dim = expected_shape[1];
    if (dst_shape[0] != in_dim) return error.TensorShapeMismatch;
    if (col_offset + out_dim > dst_shape[1]) return error.TensorShapeMismatch;
    if (dst.len != dst_shape[0] * dst_shape[1]) return error.TensorSizeMismatch;

    const dst_stride = dst_shape[1];
    switch (view.info.dtype) {
        .f32 => {
            const src = std.mem.bytesAsSlice(f32, view.data);
            if (src.len != view.info.shape[0] * view.info.shape[1]) return error.TensorSizeMismatch;

            var o: usize = 0;
            while (o < out_dim) : (o += 1) {
                var i: usize = 0;
                while (i < in_dim) : (i += 1) {
                    dst[i * dst_stride + (col_offset + o)] = src[o * in_dim + i];
                }
            }
        },
        .bf16 => {
            const src = std.mem.bytesAsSlice(u16, view.data);
            if (src.len != view.info.shape[0] * view.info.shape[1]) return error.TensorSizeMismatch;

            var o: usize = 0;
            while (o < out_dim) : (o += 1) {
                var i: usize = 0;
                while (i < in_dim) : (i += 1) {
                    dst[i * dst_stride + (col_offset + o)] = bf16_to_f32(src[o * in_dim + i]);
                }
            }
        },
        else => return error.TensorDtypeMismatch,
    }
}

fn transpose_vocab_hidden(src: []const f32, dst: []f32, src_shape: []const usize, dst_shape: []const usize) !void {
    if (src_shape.len != 2 or dst_shape.len != 2) return error.TensorShapeMismatch;
    const vocab = src_shape[0];
    const hidden = src_shape[1];
    if (dst_shape[0] != hidden or dst_shape[1] != vocab) return error.TensorShapeMismatch;

    var v: usize = 0;
    while (v < vocab) : (v += 1) {
        var h: usize = 0;
        while (h < hidden) : (h += 1) {
            dst[h * vocab + v] = src[v * hidden + h];
        }
    }
}

fn transpose_vocab_hidden_bf16(src: []const u16, dst: []u16, src_shape: []const usize, dst_shape: []const usize) !void {
    if (src_shape.len != 2 or dst_shape.len != 2) return error.TensorShapeMismatch;
    const vocab = src_shape[0];
    const hidden = src_shape[1];
    if (dst_shape[0] != hidden or dst_shape[1] != vocab) return error.TensorShapeMismatch;

    var v: usize = 0;
    while (v < vocab) : (v += 1) {
        var h: usize = 0;
        while (h < hidden) : (h += 1) {
            dst[h * vocab + v] = src[v * hidden + h];
        }
    }
}

fn bf16_to_f32(val: u16) f32 {
    const bits: u32 = @as(u32, val) << 16;
    return @bitCast(bits);
}

fn bytes_to_mb(value: i64) f64 {
    return @as(f64, @floatFromInt(value)) / (1024.0 * 1024.0);
}

fn log_compiled_memory_stats(backend_handle: anytype, exe: *zg.backend.pjrt.LoadedExecutable) void {
    const stats = backend_handle.executable_memory_stats(exe) catch |err| switch (err) {
        error.Unimplemented, error.FunctionNotAvailable => return,
        else => {
            std.log.warn("llama-ft-demo: compiled memory stats unavailable ({s})", .{@errorName(err)});
            return;
        },
    };

    std.log.info(
        "llama-ft-demo compiled memory: peak={d:.1}MB temp={d:.1}MB args={d:.1}MB outputs={d:.1}MB",
        .{
            bytes_to_mb(stats.peak_memory_in_bytes),
            bytes_to_mb(stats.temp_size_in_bytes),
            bytes_to_mb(stats.argument_size_in_bytes),
            bytes_to_mb(stats.output_size_in_bytes),
        },
    );
}

fn log_device_memory_stats(backend_handle: anytype, device: *const zg.backend.pjrt.Device) void {
    const stats = backend_handle.device_memory_stats(device) catch |err| switch (err) {
        error.Unimplemented, error.FunctionNotAvailable => return,
        else => {
            std.log.warn("llama-ft-demo: device memory stats unavailable ({s})", .{@errorName(err)});
            return;
        },
    };

    const peak = stats.peak_bytes_in_use orelse stats.bytes_in_use;
    const pool_peak = stats.peak_pool_bytes orelse stats.pool_bytes orelse 0;
    std.log.info(
        "llama-ft-demo device memory: in_use={d:.1}MB peak={d:.1}MB pool={d:.1}MB pool_peak={d:.1}MB",
        .{
            bytes_to_mb(stats.bytes_in_use),
            bytes_to_mb(peak),
            bytes_to_mb(stats.pool_bytes orelse 0),
            bytes_to_mb(pool_peak),
        },
    );
}

fn host_dtype_for(dtype: zg.pr.DType) zg.utils.DType {
    return switch (dtype) {
        .bf16 => .bf16,
        .f32 => .f32,
        .f64 => .f64,
        .i32 => .i32,
        .i64 => .i64,
        .u32 => .u32,
        .u64 => .u64,
        .bool => .i32,
    };
}

fn f32_to_bf16(val: f32) u16 {
    const bits: u32 = @bitCast(val);
    return @intCast(bits >> 16);
}

fn fill_pattern(slice: []f32, scale: f32, offset: f32) void {
    for (slice, 0..) |*v, i| {
        const base = @as(f32, @floatFromInt(i % 1024));
        v.* = offset + scale * base;
    }
}

fn fill_pattern_bf16(slice: []u16, scale: f32, offset: f32) void {
    for (slice, 0..) |*v, i| {
        const base = @as(f32, @floatFromInt(i % 1024));
        v.* = f32_to_bf16(offset + scale * base);
    }
}

fn fill_i32_tokens(out: []i32, tokens: []const usize) void {
    @memset(out, 0);
    const count = @min(out.len, tokens.len);
    for (tokens[0..count], 0..) |t, i| {
        out[i] = @intCast(t);
    }
}

fn fill_i32_tokens_batched(out: []i32, batch: usize, seq: usize, tokens: []const usize) void {
    @memset(out, 0);
    if (batch == 0 or seq == 0) return;
    if (out.len != batch * seq) return;

    var b: usize = 0;
    while (b < batch) : (b += 1) {
        const row = out[b * seq .. (b + 1) * seq];
        const count = @min(row.len, tokens.len);
        for (tokens[0..count], 0..) |t, i| {
            row[i] = @intCast(t);
        }
    }
}

fn fill_attention_mask(out: []f32, active_len: usize) void {
    @memset(out, 0);
    const count = @min(out.len, active_len);
    var i: usize = 0;
    while (i < count) : (i += 1) {
        out[i] = 1.0;
    }
}

fn fill_attention_mask_batched(out: []f32, batch: usize, seq: usize, active_len: usize) void {
    @memset(out, 0);
    if (batch == 0 or seq == 0) return;
    if (out.len != batch * seq) return;

    const count = @min(seq, active_len);
    var b: usize = 0;
    while (b < batch) : (b += 1) {
        var i: usize = 0;
        while (i < count) : (i += 1) {
            out[b * seq + i] = 1.0;
        }
    }
}

fn fill_attention_mask_bf16(out: []u16, active_len: usize) void {
    @memset(out, 0);
    const one = f32_to_bf16(1.0);
    const count = @min(out.len, active_len);
    var i: usize = 0;
    while (i < count) : (i += 1) {
        out[i] = one;
    }
}

fn fill_attention_mask_bf16_batched(out: []u16, batch: usize, seq: usize, active_len: usize) void {
    @memset(out, 0);
    if (batch == 0 or seq == 0) return;
    if (out.len != batch * seq) return;

    const one = f32_to_bf16(1.0);
    const count = @min(seq, active_len);
    var b: usize = 0;
    while (b < batch) : (b += 1) {
        var i: usize = 0;
        while (i < count) : (i += 1) {
            out[b * seq + i] = one;
        }
    }
}

fn fill_causal_mask(out: []f32, seq: usize) void {
    @memset(out, 0);
    var i: usize = 0;
    while (i < seq) : (i += 1) {
        var j: usize = 0;
        while (j < seq) : (j += 1) {
            if (j <= i) {
                out[i * seq + j] = 1.0;
            }
        }
    }
}

fn fill_causal_mask_bf16(out: []u16, seq: usize) void {
    @memset(out, 0);
    const one = f32_to_bf16(1.0);
    var i: usize = 0;
    while (i < seq) : (i += 1) {
        var j: usize = 0;
        while (j < seq) : (j += 1) {
            if (j <= i) {
                out[i * seq + j] = one;
            }
        }
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

fn fill_rope_tables(out_sin: []f32, out_cos: []f32, seq: usize, head_dim: usize) void {
    const half = head_dim / 2;
    if (out_sin.len != seq * half or out_cos.len != seq * half) return;

    const base: f32 = 500000.0;
    var inv_freq: [32]f32 = undefined;
    for (0..half) |j| {
        const exp = @as(f32, @floatFromInt(2 * j)) / @as(f32, @floatFromInt(head_dim));
        inv_freq[j] = 1.0 / std.math.pow(f32, base, exp);
    }
    llama3_rope_freq_correction(inv_freq[0..half]);

    for (0..seq) |i| {
        for (0..half) |j| {
            const theta = @as(f32, @floatFromInt(i)) * inv_freq[j];
            out_sin[i * half + j] = @sin(theta);
            out_cos[i * half + j] = @cos(theta);
        }
    }
}

fn fill_rope_tables_bf16(out_sin: []u16, out_cos: []u16, seq: usize, head_dim: usize) void {
    const half = head_dim / 2;
    if (out_sin.len != seq * half or out_cos.len != seq * half) return;

    const base: f32 = 500000.0;
    var inv_freq: [32]f32 = undefined;
    for (0..half) |j| {
        const exp = @as(f32, @floatFromInt(2 * j)) / @as(f32, @floatFromInt(head_dim));
        inv_freq[j] = 1.0 / std.math.pow(f32, base, exp);
    }
    llama3_rope_freq_correction(inv_freq[0..half]);

    for (0..seq) |i| {
        for (0..half) |j| {
            const theta = @as(f32, @floatFromInt(i)) * inv_freq[j];
            out_sin[i * half + j] = f32_to_bf16(@sin(theta));
            out_cos[i * half + j] = f32_to_bf16(@cos(theta));
        }
    }
}
