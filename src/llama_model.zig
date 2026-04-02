const std = @import("std");
const zg = @import("zigrad");
const Tensor = zg.Tensor;

pub const LlamaWeights = struct {
    w_emb: Tensor,
    w_out: Tensor,
    norm: Tensor,
    layers: []const LayerWeights,
};

pub const LayerWeights = struct {
    input_norm: Tensor,
    post_norm: Tensor,
    qkv_proj: Tensor,
    o_proj: Tensor,
    gate_proj: Tensor,
    up_proj: Tensor,
    down_proj: Tensor,
};

pub const ForwardOptions = struct {
    kernelize_provider: ?[]const u8 = null,
};

pub fn forward(
    tokens: Tensor,
    mask: Tensor,
    attention_mask: Tensor,
    sin: Tensor,
    cos: Tensor,
    weights: LlamaWeights,
    eps: f32,
    opts: ForwardOptions,
) !Tensor {
    std.debug.assert(tokens.dims().len == 2);
    std.debug.assert(weights.w_emb.dims().len == 2);
    std.debug.assert(weights.w_out.dims().len == 2);
    const batch_size = tokens.dims()[0];
    const seq = tokens.dims()[1];
    const hidden = weights.w_emb.dims()[1];
    std.debug.assert(weights.w_out.dims()[0] == hidden);

    // Build bool masks, reuse across layers
    const zero_mask = try Tensor.constant_like(mask, 0.0);
    const causal_pred = try mask.compare(zero_mask, .{ .direction = .GT, .compare_type = .FLOAT });

    const zero_attn_b = try Tensor.constant_like(attention_mask, 0.0);
    const attn_pred = try attention_mask.compare(zero_attn_b, .{ .direction = .GT, .compare_type = .FLOAT });

    // Gather embeddings w/o flattening tokens
    const token_ids = if (tokens.dtype == .i32 or tokens.dtype == .i64)
        tokens
    else
        try tokens.convert(.i32);
    const token_ids3 = try token_ids.reshape(&.{ batch_size, seq, 1 });
    const gather_params: zg.pr.GatherParams = .{
        .slice_sizes = &.{ 1, hidden },
        .offset_dims = &.{2},
        .collapsed_slice_dims = &.{0},
        .start_index_map = &.{0},
        .index_vector_dim = 2,
    };
    const x0 = try weights.w_emb.gather(token_ids3, gather_params);

    var x = x0;
    for (weights.layers, 0..) |layer, layer_idx| {
        x = try layer_forward(x, causal_pred, attn_pred, sin, cos, layer, layer_idx, eps, opts);
    }
    const final_norm = try rms_norm(x, weights.norm, eps);
    const final_norm_dot = if (final_norm.dtype == weights.w_out.dtype)
        final_norm
    else
        try final_norm.convert(weights.w_out.dtype);
    const final_flat = try final_norm_dot.reshape(&.{ batch_size * seq, hidden });
    const logits_flat = try final_flat.matmul(weights.w_out);
    return logits_flat.reshape(&.{ batch_size, seq, weights.w_out.dims()[1] });
}

fn layer_forward(
    x_in: Tensor,
    causal_pred: Tensor,
    attn_pred: Tensor,
    sin: Tensor,
    cos: Tensor,
    layer: LayerWeights,
    layer_idx: usize,
    eps: f32,
    opts: ForwardOptions,
) !Tensor {
    const x_norm = try rms_norm(x_in, layer.input_norm, eps);
    const attn_out = try self_attention(x_norm, causal_pred, attn_pred, sin, cos, layer, layer_idx, opts);
    const x1 = try x_in.add(attn_out);

    const post_norm = try rms_norm(x1, layer.post_norm, eps);
    const mlp_out = try mlp(post_norm, layer, layer_idx, opts);
    return try x1.add(mlp_out);
}

fn self_attention(
    x: Tensor,
    causal_pred: Tensor,
    attn_pred: Tensor,
    sin: Tensor,
    cos: Tensor,
    layer: LayerWeights,
    layer_idx: usize,
    opts: ForwardOptions,
) !Tensor {
    std.debug.assert(sin.dims().len == 2);
    std.debug.assert(cos.dims().len == 2);
    const rope_half = sin.dims()[1];
    std.debug.assert(cos.dims()[1] == rope_half);
    const head_dim = rope_half * 2;

    std.debug.assert(x.dims().len == 3);
    const batch_size = x.dims()[0];
    const seq = x.dims()[1];
    const hidden = x.dims()[2];
    std.debug.assert(layer.qkv_proj.dims().len == 2);
    std.debug.assert(layer.qkv_proj.dims()[0] == hidden);
    std.debug.assert(@rem(hidden, head_dim) == 0);
    const n_heads = @divExact(hidden, head_dim);
    const qkv_cols = layer.qkv_proj.dims()[1];
    std.debug.assert(@rem(qkv_cols, head_dim) == 0);
    const total_heads = @divExact(qkv_cols, head_dim);
    std.debug.assert(total_heads >= n_heads);
    std.debug.assert(@rem(total_heads - n_heads, 2) == 0);
    const n_kv = @divExact(total_heads - n_heads, 2);
    std.debug.assert(n_kv > 0);
    std.debug.assert(@rem(n_heads, n_kv) == 0);

    const x_dot = if (x.dtype == layer.qkv_proj.dtype)
        x
    else
        try x.convert(layer.qkv_proj.dtype);
    const x_flat = try x_dot.reshape(&.{ batch_size * seq, hidden });
    const qkv_region_name = try std.fmt.allocPrint(x.mode.traced.builder.program.allocator(), "llama_l{d}_attn_qkv", .{layer_idx});
    const qkv_flat = try matmul_with_optional_kernel_region(x_flat, layer.qkv_proj, opts.kernelize_provider, qkv_region_name);
    const qkv = try qkv_flat.reshape(&.{ batch_size, seq, total_heads * head_dim });
    // QKV layout is [Q heads | K heads | V heads].
    const qkv4 = try qkv.reshape(&.{ batch_size, seq, total_heads, head_dim });
    const q4 = try qkv4.slice(&.{ 0, 0, 0, 0 }, &.{ batch_size, seq, n_heads, head_dim }, &.{ 1, 1, 1, 1 });
    const k4 = try qkv4.slice(&.{ 0, 0, n_heads, 0 }, &.{ batch_size, seq, n_heads + n_kv, head_dim }, &.{ 1, 1, 1, 1 });
    const v4 = try qkv4.slice(&.{ 0, 0, n_heads + n_kv, 0 }, &.{ batch_size, seq, total_heads, head_dim }, &.{ 1, 1, 1, 1 });

    const q = try q4.reshape(&.{ batch_size, seq, n_heads, head_dim });
    const k = try k4.reshape(&.{ batch_size, seq, n_kv, head_dim });
    const v = try v4.reshape(&.{ batch_size, seq, n_kv, head_dim });

    const q_rot = try apply_rope_bshd(q, sin, cos);
    const k_rot = try apply_rope_bshd(k, sin, cos);

    const k_rep = try repeat_kv_bshd(k_rot, @divExact(n_heads, n_kv));
    const v_rep = try repeat_kv_bshd(v, @divExact(n_heads, n_kv));

    // scores: [B,H,S,S]
    // q_rot: [B,S,H,D], k_rep: [B,S,H,D]
    const scores = try q_rot.dot_general(k_rep, .{
        .lhs_batch_dims = &.{ 0, 2 },
        .rhs_batch_dims = &.{ 0, 2 },
        .lhs_contracting_dims = &.{3},
        .rhs_contracting_dims = &.{3},
    });

    const scores_f32 = if (scores.dtype == .bf16) try scores.convert(.f32) else scores;

    const scale = 1.0 / std.math.sqrt(@as(f32, @floatFromInt(head_dim)));
    const scaled = try scores_f32.mul(try Tensor.constant_like(scores_f32, scale));

    const masked = try apply_attention_masks(scaled, causal_pred, attn_pred);

    // Keep softmax math in f32 (JAX default) even when model dtype is bf16.
    const attn_f32 = try softmax_last_dim_accum_f32(masked);
    const attn = if (attn_f32.dtype == v_rep.dtype) attn_f32 else try attn_f32.convert(v_rep.dtype);

    // out_bhsd: [B,H,S,D]
    // attn: [B,H,S,S], v_rep: [B,S,H,D]
    const out_bhsd = try attn.dot_general(v_rep, .{
        .lhs_batch_dims = &.{ 0, 1 },
        .rhs_batch_dims = &.{ 0, 2 },
        .lhs_contracting_dims = &.{3},
        .rhs_contracting_dims = &.{1},
    });

    const out_bshd = try out_bhsd.transpose(&.{ 0, 2, 1, 3 });

    // Keep [B,S,H,D] and contract over [H,D] to avoid flattening.
    const out_dot = if (out_bshd.dtype == layer.o_proj.dtype)
        out_bshd
    else
        try out_bshd.convert(layer.o_proj.dtype);
    const out_flat = try out_dot.reshape(&.{ batch_size * seq, hidden });
    const o_region_name = try std.fmt.allocPrint(x.mode.traced.builder.program.allocator(), "llama_l{d}_attn_o", .{layer_idx});
    const proj_flat = try matmul_with_optional_kernel_region(out_flat, layer.o_proj, opts.kernelize_provider, o_region_name);
    return proj_flat.reshape(&.{ batch_size, seq, hidden });
}

/// Apply rotary position embeddings using split-half layout (LLaMA 3 convention).
/// x1 = x[:,:,:,:half], x2 = x[:,:,:,half:]
/// result = concat(x1*cos - x2*sin, x2*cos + x1*sin, dim=3)
fn apply_rope_bshd(x: Tensor, sin: Tensor, cos: Tensor) !Tensor {
    std.debug.assert(x.dims().len == 4);
    std.debug.assert(sin.dims().len == 2);
    std.debug.assert(cos.dims().len == 2);

    const batch_size = x.dims()[0];
    const seq = x.dims()[1];
    const heads = x.dims()[2];
    const head_dim = x.dims()[3];
    const half = @divExact(head_dim, 2);
    std.debug.assert(sin.dims()[0] == seq);
    std.debug.assert(sin.dims()[1] == half);
    std.debug.assert(cos.dims()[0] == seq);
    std.debug.assert(cos.dims()[1] == half);

    // split-half: x1 = first half of head_dim, x2 = second half
    const x1 = try x.slice(&.{ 0, 0, 0, 0 }, &.{ batch_size, seq, heads, half }, &.{ 1, 1, 1, 1 });
    const x2 = try x.slice(&.{ 0, 0, 0, half }, &.{ batch_size, seq, heads, head_dim }, &.{ 1, 1, 1, 1 });

    // broadcast sin/cos from [S, half] to [B, S, H, half]
    const sin_b = try sin.broadcast_in_dim(&.{ batch_size, seq, heads, half }, &.{ 1, 3 });
    const cos_b = try cos.broadcast_in_dim(&.{ batch_size, seq, heads, half }, &.{ 1, 3 });

    // y1 = x1 * cos - x2 * sin
    const y1 = try (try x1.mul(cos_b)).sub(try x2.mul(sin_b));
    // y2 = x2 * cos + x1 * sin
    const y2 = try (try x2.mul(cos_b)).add(try x1.mul(sin_b));

    return y1.concatenate(&.{y2}, 3);
}

fn apply_attention_masks(scores: Tensor, causal_pred: Tensor, attn_pred: Tensor) !Tensor {
    const neg_inf: f64 = -std.math.inf(f64);
    const neg_b = try Tensor.constant_like(scores, neg_inf);

    if (scores.dims().len == 3) {
        // scores: [BH, S, S], causal_pred: [S,S], attn_pred: [BH,S]
        const causal_b = try causal_pred.broadcast_in_dim(scores.dims(), &.{ 1, 2 });
        const masked_causal = try scores.select(causal_b, neg_b);
        const attn_b = try attn_pred.broadcast_in_dim(scores.dims(), &.{ 0, 2 });
        return masked_causal.select(attn_b, neg_b);
    }

    std.debug.assert(scores.dims().len == 4);
    // scores: [B,H,S,S], causal_pred: [S,S], attn_pred: [B,S]
    const causal_b = try causal_pred.broadcast_in_dim(scores.dims(), &.{ 2, 3 });
    const masked_causal = try scores.select(causal_b, neg_b);
    const attn_b = try attn_pred.broadcast_in_dim(scores.dims(), &.{ 0, 3 });
    return masked_causal.select(attn_b, neg_b);
}

fn repeat_kv(x: Tensor, repeat: i64) !Tensor {
    const seq = x.dims()[0];
    const n_kv = x.dims()[1];
    const head_dim = x.dims()[2];
    const out = try x.broadcast_in_dim(&.{ seq, n_kv, repeat, head_dim }, &.{ 0, 1, 3 });
    return out.reshape(&.{ seq, n_kv * repeat, head_dim });
}

fn repeat_kv_bshd(x: Tensor, repeat: i64) !Tensor {
    std.debug.assert(x.dims().len == 4);
    const batch_size = x.dims()[0];
    const seq = x.dims()[1];
    const n_kv = x.dims()[2];
    const head_dim = x.dims()[3];
    const out = try x.broadcast_in_dim(&.{ batch_size, seq, n_kv, repeat, head_dim }, &.{ 0, 1, 2, 4 });
    return out.reshape(&.{ batch_size, seq, n_kv * repeat, head_dim });
}

fn softmax_last_dim(x: Tensor) !Tensor {
    const rank = x.dims().len;
    const axis: i64 = @intCast(rank - 1);
    const max = try x.reduce_max(&.{axis});

    std.debug.assert(rank <= 4);
    var bd_buf: [4]i64 = .{ 0, 1, 2, 3 };
    const bd = bd_buf[0 .. rank - 1];
    const max_b = try max.broadcast_in_dim(x.dims(), bd);

    const shifted = try x.sub(max_b);
    const exp = try shifted.exp();
    const sum = try exp.reduce_sum(&.{axis});
    const sum_b = try sum.broadcast_in_dim(x.dims(), bd);
    return exp.div(sum_b);
}

// TODO: clean this up, actually a lot of this code needs clean up and reveals some gaps in the UX (tensor api surface)
fn softmax_last_dim_accum_f32(x: Tensor) !Tensor {
    if (x.dtype == .f32) return softmax_last_dim(x);
    if (x.dtype != .bf16) return softmax_last_dim(x);

    const x_f32 = try x.convert(.f32);
    const y_f32 = try softmax_last_dim(x_f32);
    return y_f32.convert(.bf16);
}

fn mlp(x: Tensor, layer: LayerWeights, layer_idx: usize, opts: ForwardOptions) !Tensor {
    const a = x.mode.traced.builder.program.allocator();
    const provider = opts.kernelize_provider;

    if (x.dims().len == 3) {
        const batch_size = x.dims()[0];
        const seq = x.dims()[1];
        const hidden = x.dims()[2];
        const x_dot = if (x.dtype == layer.gate_proj.dtype)
            x
        else
            try x.convert(layer.gate_proj.dtype);
        const x_flat = try x_dot.reshape(&.{ batch_size * seq, hidden });
        const gate = try matmul_with_optional_kernel_region(x_flat, layer.gate_proj, provider, try std.fmt.allocPrint(a, "llama_l{d}_mlp_gate", .{layer_idx}));
        const up = try matmul_with_optional_kernel_region(x_flat, layer.up_proj, provider, try std.fmt.allocPrint(a, "llama_l{d}_mlp_up", .{layer_idx}));
        const act = try silu_like(gate);
        const fused = try act.mul(up);
        const fused_dot = if (fused.dtype == layer.down_proj.dtype)
            fused
        else
            try fused.convert(layer.down_proj.dtype);
        const down_flat = try matmul_with_optional_kernel_region(fused_dot, layer.down_proj, provider, try std.fmt.allocPrint(a, "llama_l{d}_mlp_down", .{layer_idx}));
        return down_flat.reshape(&.{ batch_size, seq, hidden });
    }

    std.debug.assert(x.dims().len == 2);
    const gate = try matmul_with_optional_kernel_region(x, layer.gate_proj, provider, try std.fmt.allocPrint(a, "llama_l{d}_mlp_gate", .{layer_idx}));
    const up = try matmul_with_optional_kernel_region(x, layer.up_proj, provider, try std.fmt.allocPrint(a, "llama_l{d}_mlp_up", .{layer_idx}));
    const act = try silu_like(gate);
    const fused = try act.mul(up);
    return matmul_with_optional_kernel_region(fused, layer.down_proj, provider, try std.fmt.allocPrint(a, "llama_l{d}_mlp_down", .{layer_idx}));
}

fn matmul_with_optional_kernel_region(
    lhs: Tensor,
    rhs: Tensor,
    kernelize_provider: ?[]const u8,
    region_name: []const u8,
) !Tensor {
    if (kernelize_provider) |provider_name| {
        try lhs.mode.traced.builder.push_region(region_name, .{ .kernelize = provider_name });
        const out = lhs.matmul(rhs) catch |err| {
            lhs.mode.traced.builder.pop_region() catch {};
            return err;
        };
        try lhs.mode.traced.builder.pop_region();
        return out;
    }
    return lhs.matmul(rhs);
}

// TODO: this is hard to look at, really needs polish after we bring tensor API along.
fn silu_like(x: Tensor) !Tensor {
    // Implement silu(x) = x / (1 + exp(-x))
    // For bf16, follow a JAX-like lowering: compute exp in bf16, accumulate the rest in f32
    if (x.dtype != .bf16 and x.dtype != .f32) {
        // fallback to existing logistic lowering for other dtypes
        return x.mul(try x.logistic());
    }

    const x_f32 = if (x.dtype == .f32) x else try x.convert(.f32);

    const zero_b = try Tensor.constant_like(x_f32, 0.0);
    const neg = try zero_b.sub(x_f32);

    const exp_in = if (x.dtype == .bf16) try neg.convert(.bf16) else neg;
    const exp = try exp_in.exp();
    const exp_f32 = if (exp.dtype == .f32) exp else try exp.convert(.f32);

    const one_b = try Tensor.constant_like(x_f32, 1.0);
    const denom = try exp_f32.add(one_b);
    const inv = try one_b.div(denom);
    const y_f32 = try x_f32.mul(inv);

    return if (x.dtype == .bf16) try y_f32.convert(.bf16) else y_f32;
}

// TODO: see above comment
pub fn rms_norm(x: Tensor, weight: Tensor, eps: f32) !Tensor {
    if (x.dims().len == 3) {
        const h = x.dims()[2];
        const orig_dtype = x.dtype;
        const x_f32 = if (orig_dtype == .f32) x else try x.convert(.f32);
        const w_f32 = if (weight.dtype == .f32) weight else try weight.convert(.f32);

        const hidden_f: f32 = @floatFromInt(h);

        const x_sq = try x_f32.mul(x_f32);
        const sum = try x_sq.reduce_sum(&.{2});
        const mean = try sum.mul(try Tensor.constant_like(sum, 1.0 / hidden_f));
        const denom = try mean.add(try Tensor.constant_like(mean, eps));
        const inv = try denom.rsqrt();
        const inv_b = try inv.broadcast_in_dim(x_f32.dims(), &.{ 0, 1 });
        const normed = try x_f32.mul(inv_b);

        const w_b = try w_f32.broadcast_in_dim(x_f32.dims(), &.{2});
        const y_f32 = try normed.mul(w_b);
        return if (orig_dtype == .f32) y_f32 else y_f32.convert(orig_dtype);
    }

    std.debug.assert(x.dims().len == 2);
    const orig_dtype = x.dtype;
    const x_f32 = if (orig_dtype == .f32) x else try x.convert(.f32);
    const w_f32 = if (weight.dtype == .f32) weight else try weight.convert(.f32);

    const normed = try rms_norm_no_weight_f32(x_f32, eps);
    const w_b = try w_f32.broadcast_in_dim(normed.dims(), &.{1});
    const y_f32 = try normed.mul(w_b);
    return if (orig_dtype == .f32) y_f32 else y_f32.convert(orig_dtype);
}

fn rms_norm_no_weight_f32(x: Tensor, eps: f32) !Tensor {
    std.debug.assert(x.dtype == .f32);

    const hidden = x.dims()[1];
    const hidden_f: f32 = @floatFromInt(hidden);

    const x_sq = try x.mul(x);
    const sum = try x_sq.reduce_sum(&.{1});
    const mean = try sum.mul(try Tensor.constant_like(sum, 1.0 / hidden_f));
    const denom = try mean.add(try Tensor.constant_like(mean, eps));
    const inv = try denom.rsqrt();
    const inv_b = try inv.broadcast_in_dim(x.dims(), &.{0});
    return x.mul(inv_b);
}
