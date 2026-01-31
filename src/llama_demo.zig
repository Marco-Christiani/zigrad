const std = @import("std");
const zg = @import("zigrad");
const stz = @import("safetensors_zg");
const llama_model = @import("llama_model.zig");

const num_layers: usize = 16;

pub fn run_llama_ft_demo(
    allocator: std.mem.Allocator,
    plugin_path: []const u8,
    dump_pr: ?*zg.pipeline.DumpConfig,
    dump_mlir: ?*zg.pipeline.DumpConfig,
    warmup_steps: usize,
    steps: usize,
    quiet: bool,
) !void {
    const TensorSpec = zg.frontend.TensorSpec;

    const LayerSpec = struct {
        input_norm: TensorSpec,
        post_norm: TensorSpec,
        q_proj: TensorSpec,
        k_proj: TensorSpec,
        v_proj: TensorSpec,
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
        y: TensorSpec,
        mask: TensorSpec,
        sin: TensorSpec,
        cos: TensorSpec,
    };

    const LossFn = struct {
        fn call(params: anytype, batch: anytype) !zg.frontend.Tensor {
            const seq: usize = 4;
            const vocab: usize = 128256;

            var layers: [num_layers]llama_model.LayerWeights = undefined;
            inline for (0..num_layers) |idx| {
                const p = params.layers[idx];
                layers[idx] = .{
                    .input_norm = p.input_norm,
                    .post_norm = p.post_norm,
                    .q_proj = p.q_proj,
                    .k_proj = p.k_proj,
                    .v_proj = p.v_proj,
                    .o_proj = p.o_proj,
                    .gate_proj = p.gate_proj,
                    .up_proj = p.up_proj,
                    .down_proj = p.down_proj,
                };
            }
            const logits = try llama_model.forward(batch.x, batch.mask, batch.sin, batch.cos, .{
                .w_emb = params.w_emb,
                .w_out = params.w_out,
                .norm = params.norm,
                .layers = layers[0..],
            }, 1e-5);

            const max_logits = try logits.reduce_max(&.{1});
            const max_b = try max_logits.broadcast_in_dim(&.{ seq, vocab }, &.{0});
            const shifted = try logits.sub(max_b);
            const exp_logits = try shifted.exp();
            const sum_exp = try exp_logits.reduce_sum(&.{1});
            const log_sum = try sum_exp.log();
            const log_sum_b = try log_sum.broadcast_in_dim(&.{ seq, vocab }, &.{0});
            const log_softmax = try shifted.sub(log_sum_b);

            const y_log = try batch.y.mul(log_softmax);
            const loss_per = try y_log.reduce_sum(&.{1});

            const neg = try log_softmax.builder.scalar_literal(.{ .f32 = -1.0 });
            const neg_b = try neg.broadcast_in_dim(&.{seq}, &.{});
            const neg_loss = try loss_per.mul(neg_b);
            return try neg_loss.reduce_sum(&.{0});
        }
    };

    const seq: usize = 4;
    const vocab: usize = 128256;
    const hidden: usize = 2048;

    var layers_spec: [num_layers]LayerSpec = undefined;
    inline for (0..num_layers) |i| {
        layers_spec[i] = .{
            .input_norm = .{ .dtype = .f32, .dims = &.{hidden} },
            .post_norm = .{ .dtype = .f32, .dims = &.{hidden} },
            .q_proj = .{ .dtype = .f32, .dims = &.{ hidden, hidden } },
            .k_proj = .{ .dtype = .f32, .dims = &.{ hidden, 512 } },
            .v_proj = .{ .dtype = .f32, .dims = &.{ hidden, 512 } },
            .o_proj = .{ .dtype = .f32, .dims = &.{ hidden, hidden } },
            .gate_proj = .{ .dtype = .f32, .dims = &.{ hidden, 8192 } },
            .up_proj = .{ .dtype = .f32, .dims = &.{ hidden, 8192 } },
            .down_proj = .{ .dtype = .f32, .dims = &.{ 8192, hidden } },
        };
    }

    const params_spec = ParamsSpec{
        .w_emb = .{ .dtype = .f32, .dims = &.{ vocab, hidden } },
        .w_out = .{ .dtype = .f32, .dims = &.{ hidden, vocab } },
        .norm = .{ .dtype = .f32, .dims = &.{hidden} },
        .layers = layers_spec,
    };
    const batch_spec = BatchSpec{
        .x = .{ .dtype = .i32, .dims = &.{seq} },
        .y = .{ .dtype = .f32, .dims = &.{ seq, vocab } },
        .mask = .{ .dtype = .i32, .dims = &.{ seq, seq } },
        .sin = .{ .dtype = .f32, .dims = &.{ seq, 32 } },
        .cos = .{ .dtype = .f32, .dims = &.{ seq, 32 } },
    };
    const inputs_spec = .{ params_spec, batch_spec };

    var compile_cfg = zg.frontend.CompileConfig{
        .entry_name = "llama_ft_step",
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

    const param_count = 3 + num_layers * 9;
    const train_mode = blk: {
        const env = std.process.getEnvVarOwned(allocator, "ZG_LLAMA_TRAIN") catch null;
        defer if (env) |v| allocator.free(v);
        break :blk env != null and std.mem.eql(u8, env.?, "1");
    };

    var compiled = if (train_mode)
        try zg.frontend.compile_train_step(allocator, &backend_handle, device, LossFn.call, inputs_spec, param_count, 1e-4, compile_cfg)
    else
        try zg.frontend.compile_forward(allocator, &backend_handle, device, LossFn.call, inputs_spec, compile_cfg);
    defer compiled.deinit();

    const shape_w_emb = zg.utils.Shape{ .dims = &.{ vocab, hidden } };
    const shape_w_out = zg.utils.Shape{ .dims = &.{ hidden, vocab } };
    const shape_norm = zg.utils.Shape{ .dims = &.{hidden} };
    const shape_layer_norm = zg.utils.Shape{ .dims = &.{hidden} };
    const shape_q_proj = zg.utils.Shape{ .dims = &.{ hidden, hidden } };
    const shape_kv_proj = zg.utils.Shape{ .dims = &.{ hidden, 512 } };
    const shape_o_proj = zg.utils.Shape{ .dims = &.{ hidden, hidden } };
    const shape_gate_proj = zg.utils.Shape{ .dims = &.{ hidden, 8192 } };
    const shape_up_proj = zg.utils.Shape{ .dims = &.{ hidden, 8192 } };
    const shape_down_proj = zg.utils.Shape{ .dims = &.{ 8192, hidden } };
    const shape_x = zg.utils.Shape{ .dims = &.{seq} };
    const shape_y = zg.utils.Shape{ .dims = &.{ seq, vocab } };
    const shape_mask = zg.utils.Shape{ .dims = &.{ seq, seq } };
    const shape_rot = zg.utils.Shape{ .dims = &.{ seq, 32 } };

    var host_w_emb = try zg.utils.HostBuffer.init(allocator, shape_w_emb, .f32);
    defer host_w_emb.deinit();
    var host_w_out = try zg.utils.HostBuffer.init(allocator, shape_w_out, .f32);
    defer host_w_out.deinit();
    var host_norm = try zg.utils.HostBuffer.init(allocator, shape_norm, .f32);
    defer host_norm.deinit();
    var host_layer_input_norm: [num_layers]zg.utils.HostBuffer = undefined;
    var host_layer_post_norm: [num_layers]zg.utils.HostBuffer = undefined;
    var host_q_proj: [num_layers]zg.utils.HostBuffer = undefined;
    var host_k_proj: [num_layers]zg.utils.HostBuffer = undefined;
    var host_v_proj: [num_layers]zg.utils.HostBuffer = undefined;
    var host_o_proj: [num_layers]zg.utils.HostBuffer = undefined;
    var host_gate_proj: [num_layers]zg.utils.HostBuffer = undefined;
    var host_up_proj: [num_layers]zg.utils.HostBuffer = undefined;
    var host_down_proj: [num_layers]zg.utils.HostBuffer = undefined;

    var i: usize = 0;
    while (i < num_layers) : (i += 1) {
        host_layer_input_norm[i] = try zg.utils.HostBuffer.init(allocator, shape_layer_norm, .f32);
        host_layer_post_norm[i] = try zg.utils.HostBuffer.init(allocator, shape_layer_norm, .f32);
        host_q_proj[i] = try zg.utils.HostBuffer.init(allocator, shape_q_proj, .f32);
        host_k_proj[i] = try zg.utils.HostBuffer.init(allocator, shape_kv_proj, .f32);
        host_v_proj[i] = try zg.utils.HostBuffer.init(allocator, shape_kv_proj, .f32);
        host_o_proj[i] = try zg.utils.HostBuffer.init(allocator, shape_o_proj, .f32);
        host_gate_proj[i] = try zg.utils.HostBuffer.init(allocator, shape_gate_proj, .f32);
        host_up_proj[i] = try zg.utils.HostBuffer.init(allocator, shape_up_proj, .f32);
        host_down_proj[i] = try zg.utils.HostBuffer.init(allocator, shape_down_proj, .f32);
    }
    defer {
        var j: usize = 0;
        while (j < num_layers) : (j += 1) {
            host_layer_input_norm[j].deinit();
            host_layer_post_norm[j].deinit();
            host_q_proj[j].deinit();
            host_k_proj[j].deinit();
            host_v_proj[j].deinit();
            host_o_proj[j].deinit();
            host_gate_proj[j].deinit();
            host_up_proj[j].deinit();
            host_down_proj[j].deinit();
        }
    }
    var host_x = try zg.utils.HostBuffer.init(allocator, shape_x, .i32);
    defer host_x.deinit();
    var host_y = try zg.utils.HostBuffer.init(allocator, shape_y, .f32);
    defer host_y.deinit();
    var host_mask = try zg.utils.HostBuffer.init(allocator, shape_mask, .i32);
    defer host_mask.deinit();
    var host_sin = try zg.utils.HostBuffer.init(allocator, shape_rot, .f32);
    defer host_sin.deinit();
    var host_cos = try zg.utils.HostBuffer.init(allocator, shape_rot, .f32);
    defer host_cos.deinit();

    const default_path = "./weights/llama-3.2-1b-instruct/model.safetensors";
    const weights_path = std.process.getEnvVarOwned(allocator, "ZG_LLAMA_SAFETENSORS_PATH") catch default_path;
    defer if (!std.mem.eql(u8, weights_path, default_path)) allocator.free(weights_path);

    if (try load_llama_weights(
        allocator,
        weights_path,
        host_w_emb.as_slice(f32),
        host_w_out.as_slice(f32),
        host_norm.as_slice(f32),
        host_layer_input_norm[0..],
        host_layer_post_norm[0..],
        host_q_proj[0..],
        host_k_proj[0..],
        host_v_proj[0..],
        host_o_proj[0..],
        host_gate_proj[0..],
        host_up_proj[0..],
        host_down_proj[0..],
        shape_w_emb.dims,
        shape_w_out.dims,
        shape_norm.dims,
        shape_layer_norm.dims,
        shape_q_proj.dims,
        shape_kv_proj.dims,
        shape_o_proj.dims,
        shape_gate_proj.dims,
        shape_up_proj.dims,
        shape_down_proj.dims,
    )) {
        if (!quiet) {
            std.log.info("llama-ft-demo: loaded weights from {s}", .{weights_path});
        }
    } else {
        fill_pattern(host_w_emb.as_slice(f32), 1e-3, 0.0);
        fill_pattern(host_w_out.as_slice(f32), 1e-3, 0.0);
        fill_pattern(host_norm.as_slice(f32), 1e-3, 0.0);
        var k: usize = 0;
        while (k < num_layers) : (k += 1) {
            fill_pattern(host_layer_input_norm[k].as_slice(f32), 1e-3, 0.0);
            fill_pattern(host_layer_post_norm[k].as_slice(f32), 1e-3, 0.0);
            fill_pattern(host_q_proj[k].as_slice(f32), 1e-3, 0.0);
            fill_pattern(host_k_proj[k].as_slice(f32), 1e-3, 0.0);
            fill_pattern(host_v_proj[k].as_slice(f32), 1e-3, 0.0);
            fill_pattern(host_o_proj[k].as_slice(f32), 1e-3, 0.0);
            fill_pattern(host_gate_proj[k].as_slice(f32), 1e-3, 0.0);
            fill_pattern(host_up_proj[k].as_slice(f32), 1e-3, 0.0);
            fill_pattern(host_down_proj[k].as_slice(f32), 1e-3, 0.0);
        }
        std.log.warn("llama-ft-demo: using synthetic weights (set ZG_LLAMA_SAFETENSORS_PATH)", .{});
    }

    const tokens = [_]usize{ 128000, 128009, 128001, 128008 };
    const targets = [_]usize{ 128009, 128001, 128008, 128001 };
    fill_i32_tokens(host_x.as_slice(i32), &tokens);
    fill_one_hot_seq(host_y.as_slice(f32), &targets, vocab);
    fill_causal_mask(host_mask.as_slice(i32), seq);
    fill_rope_tables(host_sin.as_slice(f32), host_cos.as_slice(f32), seq, 64);

    var total_ns: u64 = 0;

    const tmp_w_emb = try upload_host_buffer(allocator, &backend_handle, device, &host_w_emb);
    const tmp_w_out = try upload_host_buffer(allocator, &backend_handle, device, &host_w_out);
    const tmp_norm = try upload_host_buffer(allocator, &backend_handle, device, &host_norm);
    var tmp_layer_input_norm: [num_layers]zg.backend.pjrt.Buffer = undefined;
    var tmp_layer_post_norm: [num_layers]zg.backend.pjrt.Buffer = undefined;
    var tmp_q_proj: [num_layers]zg.backend.pjrt.Buffer = undefined;
    var tmp_k_proj: [num_layers]zg.backend.pjrt.Buffer = undefined;
    var tmp_v_proj: [num_layers]zg.backend.pjrt.Buffer = undefined;
    var tmp_o_proj: [num_layers]zg.backend.pjrt.Buffer = undefined;
    var tmp_gate_proj: [num_layers]zg.backend.pjrt.Buffer = undefined;
    var tmp_up_proj: [num_layers]zg.backend.pjrt.Buffer = undefined;
    var tmp_down_proj: [num_layers]zg.backend.pjrt.Buffer = undefined;

    var m: usize = 0;
    while (m < num_layers) : (m += 1) {
        tmp_layer_input_norm[m] = try upload_host_buffer(allocator, &backend_handle, device, &host_layer_input_norm[m]);
        tmp_layer_post_norm[m] = try upload_host_buffer(allocator, &backend_handle, device, &host_layer_post_norm[m]);
        tmp_q_proj[m] = try upload_host_buffer(allocator, &backend_handle, device, &host_q_proj[m]);
        tmp_k_proj[m] = try upload_host_buffer(allocator, &backend_handle, device, &host_k_proj[m]);
        tmp_v_proj[m] = try upload_host_buffer(allocator, &backend_handle, device, &host_v_proj[m]);
        tmp_o_proj[m] = try upload_host_buffer(allocator, &backend_handle, device, &host_o_proj[m]);
        tmp_gate_proj[m] = try upload_host_buffer(allocator, &backend_handle, device, &host_gate_proj[m]);
        tmp_up_proj[m] = try upload_host_buffer(allocator, &backend_handle, device, &host_up_proj[m]);
        tmp_down_proj[m] = try upload_host_buffer(allocator, &backend_handle, device, &host_down_proj[m]);
    }
    const tmp_x = try upload_host_buffer(allocator, &backend_handle, device, &host_x);
    const tmp_y = try upload_host_buffer(allocator, &backend_handle, device, &host_y);
    const tmp_mask = try upload_host_buffer(allocator, &backend_handle, device, &host_mask);
    const tmp_sin = try upload_host_buffer(allocator, &backend_handle, device, &host_sin);
    const tmp_cos = try upload_host_buffer(allocator, &backend_handle, device, &host_cos);

    var loss_host = try zg.utils.HostBuffer.init(allocator, .{ .dims = &.{} }, .f32);
    defer loss_host.deinit();

    const output_count = if (train_mode) 1 + param_count else 1;
    const api = tmp_w_emb.api;

    var input_ptrs = std.ArrayList(zg.backend.pjrt.RawBuffer).empty;
    defer input_ptrs.deinit(allocator);
    try input_ptrs.append(allocator, tmp_w_emb.pjrt_buffer);
    try input_ptrs.append(allocator, tmp_w_out.pjrt_buffer);
    try input_ptrs.append(allocator, tmp_norm.pjrt_buffer);
    var p: usize = 0;
    while (p < num_layers) : (p += 1) {
        try input_ptrs.append(allocator, tmp_layer_input_norm[p].pjrt_buffer);
        try input_ptrs.append(allocator, tmp_layer_post_norm[p].pjrt_buffer);
        try input_ptrs.append(allocator, tmp_q_proj[p].pjrt_buffer);
        try input_ptrs.append(allocator, tmp_k_proj[p].pjrt_buffer);
        try input_ptrs.append(allocator, tmp_v_proj[p].pjrt_buffer);
        try input_ptrs.append(allocator, tmp_o_proj[p].pjrt_buffer);
        try input_ptrs.append(allocator, tmp_gate_proj[p].pjrt_buffer);
        try input_ptrs.append(allocator, tmp_up_proj[p].pjrt_buffer);
        try input_ptrs.append(allocator, tmp_down_proj[p].pjrt_buffer);
    }
    try input_ptrs.append(allocator, tmp_x.pjrt_buffer);
    try input_ptrs.append(allocator, tmp_y.pjrt_buffer);
    try input_ptrs.append(allocator, tmp_mask.pjrt_buffer);
    try input_ptrs.append(allocator, tmp_sin.pjrt_buffer);
    try input_ptrs.append(allocator, tmp_cos.pjrt_buffer);

    var output_ptrs = try allocator.alloc(zg.backend.pjrt.RawBuffer, output_count);
    defer allocator.free(output_ptrs);

    const non_donatable = &.{
        @as(i64, @intCast(param_count)),
        @as(i64, @intCast(param_count + 1)),
        @as(i64, @intCast(param_count + 2)),
        @as(i64, @intCast(param_count + 3)),
        @as(i64, @intCast(param_count + 4)),
    };
    const exec_opts: zg.frontend.CompiledForward.ExecuteOptions = .{
        .non_donatable_input_indices = if (train_mode) non_donatable else null,
    };

    defer {
        for (input_ptrs.items) |raw| {
            var buf = zg.backend.pjrt.Buffer{ .api = api, .pjrt_buffer = raw };
            buf.deinit();
        }
    }

    const is_cpu = try (zg.backend.pjrt.Buffer{ .api = api, .pjrt_buffer = input_ptrs.items[0] }).is_on_cpu();

    var warmup: usize = 0;
    while (warmup < warmup_steps) : (warmup += 1) {
        const ev = try compiled.execute_into(input_ptrs.items, output_ptrs, exec_opts);

        var loss_buf = zg.backend.pjrt.Buffer{ .api = api, .pjrt_buffer = output_ptrs[0] };
        if (ev) |e| {
            var tmp = e;
            try tmp.await_();
            tmp.deinit();
        }
        if (!is_cpu and !quiet) {
            var loss_ev = try loss_buf.to_host(loss_host.data);
            try loss_ev.await_();
            loss_ev.deinit();
        }
        loss_buf.deinit();

        if (train_mode) {
            for (input_ptrs.items[0..param_count], output_ptrs[1..]) |*old, new| {
                if (new == old.*) continue;
                var buf = zg.backend.pjrt.Buffer{ .api = api, .pjrt_buffer = old.* };
                old.* = new;
                buf.deinit();
            }
        }
    }

    var step: usize = 0;
    while (step < steps) : (step += 1) {
        var timer = try std.time.Timer.start();
        const event = try compiled.execute_into(input_ptrs.items, output_ptrs, exec_opts);
        const dispatch_ns = timer.lap();

        if (event) |ev| {
            var tmp = ev;
            try tmp.await_();
            tmp.deinit();
        }
        const wait_ns = timer.lap();

        var loss_buf = zg.backend.pjrt.Buffer{ .api = api, .pjrt_buffer = output_ptrs[0] };
        const loss: ?f32 = if (quiet) null else if (is_cpu) blk: {
            const ptr: [*]const f32 = @ptrFromInt(try loss_buf.unsafe_pointer());
            break :blk ptr[0];
        } else blk: {
            var loss_ev = try loss_buf.to_host(loss_host.data);
            try loss_ev.await_();
            loss_ev.deinit();
            break :blk loss_host.as_slice(f32)[0];
        };
        const loss_read_ns = timer.lap();

        loss_buf.deinit();

        if (train_mode) {
            for (input_ptrs.items[0..param_count], output_ptrs[1..]) |*old, new| {
                if (new == old.*) continue;
                var buf = zg.backend.pjrt.Buffer{ .api = api, .pjrt_buffer = old.* };
                old.* = new;
                buf.deinit();
            }
        }

        const step_ns = dispatch_ns + wait_ns + loss_read_ns;
        total_ns += step_ns;
        const step_ms = @as(f64, @floatFromInt(step_ns)) / std.time.ns_per_ms;
        const dispatch_ms = @as(f64, @floatFromInt(dispatch_ns)) / std.time.ns_per_ms;
        const wait_ms = @as(f64, @floatFromInt(wait_ns)) / std.time.ns_per_ms;
        const loss_ms = @as(f64, @floatFromInt(loss_read_ns)) / std.time.ns_per_ms;
        if (!quiet) {
            std.log.info("llama-ft-demo step {d}: loss={d:.6} dispatch={d:.3}ms wait={d:.3}ms loss={d:.3}ms total={d:.3}ms", .{
                step, loss.?, dispatch_ms, wait_ms, loss_ms, step_ms,
            });
        }
    }

    const avg_ms = @as(f64, @floatFromInt(total_ns)) / std.time.ns_per_ms / @as(f64, @floatFromInt(steps));
    std.log.info("llama-ft-demo avg_step_ms={d:.3}", .{avg_ms});
    std.log.info("OK: llama-ft-demo executed", .{});
}

fn load_llama_weights(
    allocator: std.mem.Allocator,
    path: []const u8,
    w_emb: []f32,
    w_out: []f32,
    norm: []f32,
    layer_input_norm: []zg.utils.HostBuffer,
    layer_post_norm: []zg.utils.HostBuffer,
    q_proj: []zg.utils.HostBuffer,
    k_proj: []zg.utils.HostBuffer,
    v_proj: []zg.utils.HostBuffer,
    o_proj: []zg.utils.HostBuffer,
    gate_proj: []zg.utils.HostBuffer,
    up_proj: []zg.utils.HostBuffer,
    down_proj: []zg.utils.HostBuffer,
    shape_w_emb: []const usize,
    shape_w_out: []const usize,
    shape_norm: []const usize,
    shape_layer_norm: []const usize,
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
    try copy_tensor_to_f32(emb_view, w_emb, shape_w_emb);

    const norm_view = try st_file.get("model.norm.weight");
    try copy_tensor_to_f32(norm_view, norm, shape_norm);

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
        try copy_tensor_to_f32(layer_in, layer_input_norm[i].as_slice(f32), shape_layer_norm);
        const layer_post = try st_file.get(post_name);
        try copy_tensor_to_f32(layer_post, layer_post_norm[i].as_slice(f32), shape_layer_norm);

        const q_view = try st_file.get(q_name);
        try copy_tensor_to_f32_transposed(q_view, q_proj[i].as_slice(f32), shape_q_proj);
        const k_view = try st_file.get(k_name);
        try copy_tensor_to_f32_transposed(k_view, k_proj[i].as_slice(f32), shape_k_proj);
        const v_view = try st_file.get(v_name);
        try copy_tensor_to_f32_transposed(v_view, v_proj[i].as_slice(f32), shape_k_proj);
        const o_view = try st_file.get(o_name);
        try copy_tensor_to_f32_transposed(o_view, o_proj[i].as_slice(f32), shape_o_proj);

        const gate_view = try st_file.get(gate_name);
        try copy_tensor_to_f32_transposed(gate_view, gate_proj[i].as_slice(f32), shape_gate_proj);
        const up_view = try st_file.get(up_name);
        try copy_tensor_to_f32_transposed(up_view, up_proj[i].as_slice(f32), shape_up_proj);
        const down_view = try st_file.get(down_name);
        try copy_tensor_to_f32_transposed(down_view, down_proj[i].as_slice(f32), shape_down_proj);
    }

    const lm_view = get_optional(&st_file, "lm_head.weight") catch |err| return err;
    if (lm_view) |view| {
        try copy_tensor_to_f32_transposed(view, w_out, shape_w_out);
    } else {
        try transpose_vocab_hidden(w_emb, w_out, shape_w_emb, shape_w_out);
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

fn bf16_to_f32(val: u16) f32 {
    const bits: u32 = @as(u32, val) << 16;
    return @bitCast(bits);
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

fn fill_one_hot_seq(out: []f32, indices: []const usize, vocab: usize) void {
    @memset(out, 0);
    for (indices, 0..) |idx, i| {
        const offset = i * vocab + (idx % vocab);
        if (offset < out.len) out[offset] = 1.0;
    }
}

fn fill_i32_tokens(out: []i32, tokens: []const usize) void {
    if (out.len != tokens.len) return;
    for (tokens, 0..) |t, i| {
        out[i] = @intCast(t);
    }
}

fn fill_causal_mask(out: []i32, seq: usize) void {
    @memset(out, 0);
    var i: usize = 0;
    while (i < seq) : (i += 1) {
        var j: usize = 0;
        while (j < seq) : (j += 1) {
            if (j <= i) {
                out[i * seq + j] = 1;
            }
        }
    }
}

fn fill_rope_tables(out_sin: []f32, out_cos: []f32, seq: usize, head_dim: usize) void {
    const half = head_dim / 2;
    if (out_sin.len != seq * half or out_cos.len != seq * half) return;

    const base: f32 = 10000.0;
    var i: usize = 0;
    while (i < seq) : (i += 1) {
        var j: usize = 0;
        while (j < half) : (j += 1) {
            const exp = @as(f32, @floatFromInt(2 * j)) / @as(f32, @floatFromInt(head_dim));
            const inv = 1.0 / std.math.pow(f32, base, exp);
            const theta = @as(f32, @floatFromInt(i)) * inv;
            out_sin[i * half + j] = @sin(theta);
            out_cos[i * half + j] = @cos(theta);
        }
    }
}
