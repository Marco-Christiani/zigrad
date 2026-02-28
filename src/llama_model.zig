const std = @import("std");
const zg = @import("zigrad");
const ops = zg.pr.ops;

pub const LlamaWeights = struct {
    w_emb: zg.frontend.Tensor,
    w_out: zg.frontend.Tensor,
    norm: zg.frontend.Tensor,
    layers: []const LayerWeights,
};

pub const LayerWeights = struct {
    input_norm: zg.frontend.Tensor,
    post_norm: zg.frontend.Tensor,
    qkv_proj: zg.frontend.Tensor,
    o_proj: zg.frontend.Tensor,
    gate_proj: zg.frontend.Tensor,
    up_proj: zg.frontend.Tensor,
    down_proj: zg.frontend.Tensor,
};

pub const ForwardOptions = struct {
    kernelize_provider: ?[]const u8 = null,
};

pub fn forward(
    tokens: zg.frontend.Tensor,
    mask: zg.frontend.Tensor,
    attention_mask: zg.frontend.Tensor,
    sin: zg.frontend.Tensor,
    cos: zg.frontend.Tensor,
    weights: LlamaWeights,
    eps: f32,
) !zg.frontend.Tensor {
    return forward_with_options(tokens, mask, attention_mask, sin, cos, weights, eps, .{});
}

pub fn forward_with_options(
    tokens: zg.frontend.Tensor,
    mask: zg.frontend.Tensor,
    attention_mask: zg.frontend.Tensor,
    sin: zg.frontend.Tensor,
    cos: zg.frontend.Tensor,
    weights: LlamaWeights,
    eps: f32,
    opts: ForwardOptions,
) !zg.frontend.Tensor {
    std.debug.assert(tokens.tensor.shape.dims.len == 2);
    std.debug.assert(weights.w_emb.tensor.shape.dims.len == 2);
    std.debug.assert(weights.w_out.tensor.shape.dims.len == 2);
    const batch_size: usize = tokens.tensor.shape.dims[0];
    const seq: usize = tokens.tensor.shape.dims[1];
    const hidden: usize = weights.w_emb.tensor.shape.dims[1];
    std.debug.assert(weights.w_out.tensor.shape.dims[0] == hidden);

    // Build boolean masks once and reuse across layers.
    const zero_lit = ops.types.scalar_literal(mask.tensor.dtype, 0.0);
    const zero = try mask.builder.scalar_literal(zero_lit);
    const zero_mask = try zero.broadcast_in_dim(mask.tensor.shape.dims, &.{});
    const causal_pred = try mask.compare(zero_mask, .{ .direction = .GT, .compare_type = .FLOAT });

    const zero_attn = if (attention_mask.tensor.dtype == mask.tensor.dtype)
        zero
    else
        try attention_mask.builder.scalar_literal(ops.types.scalar_literal(attention_mask.tensor.dtype, 0.0));
    const zero_attn_b = try zero_attn.broadcast_in_dim(attention_mask.tensor.shape.dims, &.{});
    const attn_pred = try attention_mask.compare(zero_attn_b, .{ .direction = .GT, .compare_type = .FLOAT });

    // Gather embeddings without flattening tokens.
    const token_ids = if (tokens.tensor.dtype == .i32 or tokens.tensor.dtype == .i64)
        tokens
    else
        try tokens.convert(.i32);
    const token_ids3 = try token_ids.reshape(&.{ batch_size, seq, 1 });
    const hidden_i64: i64 = @intCast(hidden);
    const gather_params: zg.pr.GatherParams = .{
        .slice_sizes = &.{ 1, hidden_i64 },
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
    const final_norm_dot = if (final_norm.tensor.dtype == weights.w_out.tensor.dtype)
        final_norm
    else
        try final_norm.convert(weights.w_out.tensor.dtype);
    const final_flat = try final_norm_dot.reshape(&.{ batch_size * seq, hidden });
    const logits_flat = try final_flat.matmul(weights.w_out);
    return logits_flat.reshape(&.{ batch_size, seq, weights.w_out.tensor.shape.dims[1] });
}

fn layer_forward(
    x_in: zg.frontend.Tensor,
    causal_pred: zg.frontend.Tensor,
    attn_pred: zg.frontend.Tensor,
    sin: zg.frontend.Tensor,
    cos: zg.frontend.Tensor,
    layer: LayerWeights,
    layer_idx: usize,
    eps: f32,
    opts: ForwardOptions,
) !zg.frontend.Tensor {
    const x_norm = try rms_norm(x_in, layer.input_norm, eps);
    const attn_out = try self_attention(x_norm, causal_pred, attn_pred, sin, cos, layer, layer_idx, opts);
    const x1 = try x_in.add(attn_out);

    const post_norm = try rms_norm(x1, layer.post_norm, eps);
    const mlp_out = try mlp(post_norm, layer);
    return x1.add(mlp_out);
}

fn self_attention(
    x: zg.frontend.Tensor,
    causal_pred: zg.frontend.Tensor,
    attn_pred: zg.frontend.Tensor,
    sin: zg.frontend.Tensor,
    cos: zg.frontend.Tensor,
    layer: LayerWeights,
    layer_idx: usize,
    opts: ForwardOptions,
) !zg.frontend.Tensor {
    std.debug.assert(sin.tensor.shape.dims.len == 2);
    std.debug.assert(cos.tensor.shape.dims.len == 2);
    const rope_half: usize = sin.tensor.shape.dims[1];
    std.debug.assert(cos.tensor.shape.dims[1] == rope_half);
    const head_dim: usize = rope_half * 2;

    std.debug.assert(x.tensor.shape.dims.len == 3);
    const batch_size: usize = x.tensor.shape.dims[0];
    const seq: usize = x.tensor.shape.dims[1];
    const hidden: usize = x.tensor.shape.dims[2];
    std.debug.assert(layer.qkv_proj.tensor.shape.dims.len == 2);
    std.debug.assert(layer.qkv_proj.tensor.shape.dims[0] == hidden);
    std.debug.assert(hidden % head_dim == 0);
    const n_heads: usize = hidden / head_dim;
    const qkv_cols: usize = layer.qkv_proj.tensor.shape.dims[1];
    std.debug.assert(qkv_cols % head_dim == 0);
    const total_heads: usize = qkv_cols / head_dim;
    std.debug.assert(total_heads >= n_heads);
    std.debug.assert((total_heads - n_heads) % 2 == 0);
    const n_kv: usize = (total_heads - n_heads) / 2;
    std.debug.assert(n_kv > 0);
    std.debug.assert(n_heads % n_kv == 0);

    const x_dot = if (x.tensor.dtype == layer.qkv_proj.tensor.dtype)
        x
    else
        try x.convert(layer.qkv_proj.tensor.dtype);
    const x_flat = try x_dot.reshape(&.{ batch_size * seq, hidden });
    const qkv_provider = if (layer_idx == 0) opts.kernelize_provider else null;
    const qkv_region_name = try std.fmt.allocPrint(x.builder.program.allocator(), "llama_l{d}_attn_qkv", .{layer_idx});
    const qkv_flat = try matmul_with_optional_kernel_region(x_flat, layer.qkv_proj, qkv_provider, qkv_region_name);
    const qkv = try qkv_flat.reshape(&.{ batch_size, seq, total_heads * head_dim });
    // QKV layout is [Q heads | K heads | V heads].
    const qkv4 = try qkv.reshape(&.{ batch_size, seq, total_heads, head_dim });
    const b_i64: i64 = @intCast(batch_size);
    const s_i64: i64 = @intCast(seq);
    const heads_i64: i64 = @intCast(n_heads);
    const kv_i64: i64 = @intCast(n_kv);
    const total_i64: i64 = @intCast(total_heads);
    const head_dim_i64: i64 = @intCast(head_dim);
    const q4 = try qkv4.slice(&.{ 0, 0, 0, 0 }, &.{ b_i64, s_i64, heads_i64, head_dim_i64 }, &.{ 1, 1, 1, 1 });
    const k4 = try qkv4.slice(&.{ 0, 0, heads_i64, 0 }, &.{ b_i64, s_i64, heads_i64 + kv_i64, head_dim_i64 }, &.{ 1, 1, 1, 1 });
    const v4 = try qkv4.slice(&.{ 0, 0, heads_i64 + kv_i64, 0 }, &.{ b_i64, s_i64, total_i64, head_dim_i64 }, &.{ 1, 1, 1, 1 });

    const q = try q4.reshape(&.{ batch_size, seq, n_heads, head_dim });
    const k = try k4.reshape(&.{ batch_size, seq, n_kv, head_dim });
    const v = try v4.reshape(&.{ batch_size, seq, n_kv, head_dim });

    const q_rot = try apply_rope_bshd(q, sin, cos);
    const k_rot = try apply_rope_bshd(k, sin, cos);

    const k_rep = try repeat_kv_bshd(k_rot, n_heads / n_kv);
    const v_rep = try repeat_kv_bshd(v, n_heads / n_kv);

    // scores: [B,H,S,S]
    // q_rot: [B,S,H,D], k_rep: [B,S,H,D]
    const scores = try q_rot.dot_general(k_rep, .{
        .lhs_batch_dims = &.{ 0, 2 },
        .rhs_batch_dims = &.{ 0, 2 },
        .lhs_contracting_dims = &.{3},
        .rhs_contracting_dims = &.{3},
    });

    const scores_f32 = if (scores.tensor.dtype == .bf16) try scores.convert(.f32) else scores;

    const scale = 1.0 / std.math.sqrt(@as(f32, @floatFromInt(head_dim)));
    const scale_lit = ops.types.scalar_literal(scores_f32.tensor.dtype, scale);
    const scale_t = try scores_f32.builder.scalar_literal(scale_lit);
    const scale_b = try scale_t.broadcast_in_dim(scores_f32.tensor.shape.dims, &.{});
    const scaled = try scores_f32.mul(scale_b);

    const masked = try apply_attention_masks(scaled, causal_pred, attn_pred);

    // Keep softmax math in f32 (JAX default) even when model dtype is bf16.
    const attn_f32 = try softmax_last_dim_accum_f32(masked);
    const attn = if (attn_f32.tensor.dtype == v_rep.tensor.dtype) attn_f32 else try attn_f32.convert(v_rep.tensor.dtype);

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
    const out_dot = if (out_bshd.tensor.dtype == layer.o_proj.tensor.dtype)
        out_bshd
    else
        try out_bshd.convert(layer.o_proj.tensor.dtype);
    const out_flat = try out_dot.reshape(&.{ batch_size * seq, hidden });
    const proj_flat = try out_flat.matmul(layer.o_proj);
    return proj_flat.reshape(&.{ batch_size, seq, hidden });
}

/// Apply rotary position embeddings using split-half layout (LLaMA 3 convention).
/// x1 = x[:,:,:,:half], x2 = x[:,:,:,half:]
/// result = concat(x1*cos - x2*sin, x2*cos + x1*sin, dim=3)
fn apply_rope_bshd(x: zg.frontend.Tensor, sin: zg.frontend.Tensor, cos: zg.frontend.Tensor) !zg.frontend.Tensor {
    std.debug.assert(x.tensor.shape.dims.len == 4);
    std.debug.assert(sin.tensor.shape.dims.len == 2);
    std.debug.assert(cos.tensor.shape.dims.len == 2);

    const batch_size = x.tensor.shape.dims[0];
    const seq = x.tensor.shape.dims[1];
    const heads = x.tensor.shape.dims[2];
    const head_dim = x.tensor.shape.dims[3];
    const half = head_dim / 2;
    std.debug.assert(sin.tensor.shape.dims[0] == seq);
    std.debug.assert(sin.tensor.shape.dims[1] == half);
    std.debug.assert(cos.tensor.shape.dims[0] == seq);
    std.debug.assert(cos.tensor.shape.dims[1] == half);

    const b_i64: i64 = @intCast(batch_size);
    const s_i64: i64 = @intCast(seq);
    const h_i64: i64 = @intCast(heads);
    const half_i64: i64 = @intCast(half);

    // Split-half: x1 = first half of head_dim, x2 = second half.
    const x1 = try x.slice(&.{ 0, 0, 0, 0 }, &.{ b_i64, s_i64, h_i64, half_i64 }, &.{ 1, 1, 1, 1 });
    const x2 = try x.slice(&.{ 0, 0, 0, half_i64 }, &.{ b_i64, s_i64, h_i64, @as(i64, @intCast(head_dim)) }, &.{ 1, 1, 1, 1 });

    // Broadcast sin/cos from [S, half] to [B, S, H, half].
    const sin_b = try sin.broadcast_in_dim(&.{ batch_size, seq, heads, half }, &.{ 1, 3 });
    const cos_b = try cos.broadcast_in_dim(&.{ batch_size, seq, heads, half }, &.{ 1, 3 });

    // y1 = x1 * cos - x2 * sin
    const y1 = try (try x1.mul(cos_b)).sub(try x2.mul(sin_b));
    // y2 = x2 * cos + x1 * sin
    const y2 = try (try x2.mul(cos_b)).add(try x1.mul(sin_b));

    return y1.concatenate(&.{y2}, 3);
}

fn apply_attention_masks(scores: zg.frontend.Tensor, causal_pred: zg.frontend.Tensor, attn_pred: zg.frontend.Tensor) !zg.frontend.Tensor {
    // Use -inf (rather than a large negative constant) to match typical XLA/JAX masking.
    const neg_inf: f64 = -std.math.inf(f64);
    const neg_lit = ops.types.scalar_literal(scores.tensor.dtype, neg_inf);
    const neg = try scores.builder.scalar_literal(neg_lit);
    const neg_b = try neg.broadcast_in_dim(scores.tensor.shape.dims, &.{});

    if (scores.tensor.shape.dims.len == 3) {
        // scores: [BH, S, S], causal_pred: [S,S], attn_pred: [BH,S]
        const causal_b = try causal_pred.broadcast_in_dim(scores.tensor.shape.dims, &.{ 1, 2 });
        const masked_causal = try scores.select(causal_b, neg_b);
        const attn_b = try attn_pred.broadcast_in_dim(scores.tensor.shape.dims, &.{ 0, 2 });
        return masked_causal.select(attn_b, neg_b);
    }

    std.debug.assert(scores.tensor.shape.dims.len == 4);
    // scores: [B,H,S,S], causal_pred: [S,S], attn_pred: [B,S]
    const causal_b = try causal_pred.broadcast_in_dim(scores.tensor.shape.dims, &.{ 2, 3 });
    const masked_causal = try scores.select(causal_b, neg_b);
    const attn_b = try attn_pred.broadcast_in_dim(scores.tensor.shape.dims, &.{ 0, 3 });
    return masked_causal.select(attn_b, neg_b);
}

fn repeat_kv(x: zg.frontend.Tensor, repeat: usize) !zg.frontend.Tensor {
    const seq = x.tensor.shape.dims[0];
    const n_kv = x.tensor.shape.dims[1];
    const head_dim = x.tensor.shape.dims[2];
    const out = try x.broadcast_in_dim(&.{ seq, n_kv, repeat, head_dim }, &.{ 0, 1, 3 });
    return out.reshape(&.{ seq, n_kv * repeat, head_dim });
}

fn repeat_kv_bshd(x: zg.frontend.Tensor, repeat: usize) !zg.frontend.Tensor {
    std.debug.assert(x.tensor.shape.dims.len == 4);
    const batch_size = x.tensor.shape.dims[0];
    const seq = x.tensor.shape.dims[1];
    const n_kv = x.tensor.shape.dims[2];
    const head_dim = x.tensor.shape.dims[3];
    const out = try x.broadcast_in_dim(&.{ batch_size, seq, n_kv, repeat, head_dim }, &.{ 0, 1, 2, 4 });
    return out.reshape(&.{ batch_size, seq, n_kv * repeat, head_dim });
}

fn softmax_last_dim(x: zg.frontend.Tensor) !zg.frontend.Tensor {
    const rank = x.tensor.shape.dims.len;
    const axis: i64 = @intCast(rank - 1);
    const max = try x.reduce_max(&.{axis});

    std.debug.assert(rank <= 4);
    var bd_buf: [4]i64 = .{ 0, 1, 2, 3 };
    const bd = bd_buf[0 .. rank - 1];
    const max_b = try max.broadcast_in_dim(x.tensor.shape.dims, bd);

    const shifted = try x.sub(max_b);
    const exp = try shifted.exp();
    const sum = try exp.reduce_sum(&.{axis});
    const sum_b = try sum.broadcast_in_dim(x.tensor.shape.dims, bd);
    return exp.div(sum_b);
}

fn softmax_last_dim_accum_f32(x: zg.frontend.Tensor) !zg.frontend.Tensor {
    if (x.tensor.dtype == .f32) return softmax_last_dim(x);
    if (x.tensor.dtype != .bf16) return softmax_last_dim(x);

    const x_f32 = try x.convert(.f32);
    const y_f32 = try softmax_last_dim(x_f32);
    return y_f32.convert(.bf16);
}

fn mlp(x: zg.frontend.Tensor, layer: LayerWeights) !zg.frontend.Tensor {
    if (x.tensor.shape.dims.len == 3) {
        const batch_size = x.tensor.shape.dims[0];
        const seq = x.tensor.shape.dims[1];
        const hidden = x.tensor.shape.dims[2];
        const x_dot = if (x.tensor.dtype == layer.gate_proj.tensor.dtype)
            x
        else
            try x.convert(layer.gate_proj.tensor.dtype);
        const x_flat = try x_dot.reshape(&.{ batch_size * seq, hidden });
        const gate = try x_flat.matmul(layer.gate_proj);
        const up = try x_flat.matmul(layer.up_proj);
        const act = try silu_like(gate);
        const fused = try act.mul(up);
        const fused_dot = if (fused.tensor.dtype == layer.down_proj.tensor.dtype)
            fused
        else
            try fused.convert(layer.down_proj.tensor.dtype);
        const down_flat = try fused_dot.matmul(layer.down_proj);
        return down_flat.reshape(&.{ batch_size, seq, hidden });
    }

    std.debug.assert(x.tensor.shape.dims.len == 2);
    const gate = try x.matmul(layer.gate_proj);
    const up = try x.matmul(layer.up_proj);
    const act = try silu_like(gate);
    const fused = try act.mul(up);
    return fused.matmul(layer.down_proj);
}

fn matmul_with_optional_kernel_region(
    lhs: zg.frontend.Tensor,
    rhs: zg.frontend.Tensor,
    kernelize_provider: ?[]const u8,
    region_name: []const u8,
) !zg.frontend.Tensor {
    if (kernelize_provider) |provider_name| {
        try lhs.builder.push_region(region_name, .{ .kernelize = provider_name });
        const out = lhs.matmul(rhs) catch |err| {
            lhs.builder.pop_region() catch {};
            return err;
        };
        try lhs.builder.pop_region();
        return out;
    }
    return lhs.matmul(rhs);
}

fn silu_like(x: zg.frontend.Tensor) !zg.frontend.Tensor {
    // Implement silu(x) = x / (1 + exp(-x)).
    // For bf16, follow a JAX-like lowering: compute exp in bf16, accumulate the rest in f32.
    if (x.tensor.dtype != .bf16 and x.tensor.dtype != .f32) {
        // Fallback to existing logistic lowering for other dtypes.
        return x.mul(try x.logistic());
    }

    const x_f32 = if (x.tensor.dtype == .f32) x else try x.convert(.f32);
    const b = x_f32.builder;
    const shape = x_f32.tensor.shape.dims;

    const zero = try b.scalar_literal(ops.types.scalar_literal(.f32, 0.0));
    const zero_b = try zero.broadcast_in_dim(shape, &.{});
    const neg = try zero_b.sub(x_f32);

    const exp_in = if (x.tensor.dtype == .bf16) try neg.convert(.bf16) else neg;
    const exp = try exp_in.exp();
    const exp_f32 = if (exp.tensor.dtype == .f32) exp else try exp.convert(.f32);

    const one = try b.scalar_literal(ops.types.scalar_literal(.f32, 1.0));
    const one_b = try one.broadcast_in_dim(shape, &.{});
    const denom = try exp_f32.add(one_b);
    const inv = try one_b.div(denom);
    const y_f32 = try x_f32.mul(inv);

    return if (x.tensor.dtype == .bf16) try y_f32.convert(.bf16) else y_f32;
}

pub fn rms_norm(x: zg.frontend.Tensor, weight: zg.frontend.Tensor, eps: f32) !zg.frontend.Tensor {
    if (x.tensor.shape.dims.len == 3) {
        const h = x.tensor.shape.dims[2];
        const orig_dtype = x.tensor.dtype;
        const x_f32 = if (orig_dtype == .f32) x else try x.convert(.f32);
        const w_f32 = if (weight.tensor.dtype == .f32) weight else try weight.convert(.f32);

        const hidden_f: f32 = @floatFromInt(h);
        const mean_scale_lit = ops.types.scalar_literal(.f32, 1.0 / hidden_f);
        const mean_scale = try x_f32.builder.scalar_literal(mean_scale_lit);

        const x_sq = try x_f32.mul(x_f32);
        const sum = try x_sq.reduce_sum(&.{2});
        const mean_scale_b = try mean_scale.broadcast_in_dim(&.{ x_f32.tensor.shape.dims[0], x_f32.tensor.shape.dims[1] }, &.{});
        const mean = try sum.mul(mean_scale_b);

        const eps_lit = ops.types.scalar_literal(.f32, eps);
        const eps_tensor = try x_f32.builder.scalar_literal(eps_lit);
        const eps_b = try eps_tensor.broadcast_in_dim(&.{ x_f32.tensor.shape.dims[0], x_f32.tensor.shape.dims[1] }, &.{});
        const denom = try mean.add(eps_b);
        const inv = try denom.rsqrt();
        const inv_b = try inv.broadcast_in_dim(x_f32.tensor.shape.dims, &.{ 0, 1 });
        const normed = try x_f32.mul(inv_b);

        const w_b = try w_f32.broadcast_in_dim(x_f32.tensor.shape.dims, &.{2});
        const y_f32 = try normed.mul(w_b);
        return if (orig_dtype == .f32) y_f32 else y_f32.convert(orig_dtype);
    }

    std.debug.assert(x.tensor.shape.dims.len == 2);
    const orig_dtype = x.tensor.dtype;
    const x_f32 = if (orig_dtype == .f32) x else try x.convert(.f32);
    const w_f32 = if (weight.tensor.dtype == .f32) weight else try weight.convert(.f32);

    const normed = try rms_norm_no_weight_f32(x_f32, eps);
    const w_b = try w_f32.broadcast_in_dim(normed.tensor.shape.dims, &.{1});
    const y_f32 = try normed.mul(w_b);
    return if (orig_dtype == .f32) y_f32 else y_f32.convert(orig_dtype);
}

fn rms_norm_no_weight_f32(x: zg.frontend.Tensor, eps: f32) !zg.frontend.Tensor {
    std.debug.assert(x.tensor.dtype == .f32);

    const hidden = x.tensor.shape.dims[1];
    const hidden_f: f32 = @floatFromInt(hidden);
    const mean_scale_lit = ops.types.scalar_literal(.f32, 1.0 / hidden_f);
    const mean_scale = try x.builder.scalar_literal(mean_scale_lit);

    const x_sq = try x.mul(x);
    const sum = try x_sq.reduce_sum(&.{1});
    const mean_scale_b = try mean_scale.broadcast_in_dim(&.{x.tensor.shape.dims[0]}, &.{});
    const mean = try sum.mul(mean_scale_b);

    const eps_lit = ops.types.scalar_literal(.f32, eps);
    const eps_tensor = try x.builder.scalar_literal(eps_lit);
    const eps_b = try eps_tensor.broadcast_in_dim(&.{x.tensor.shape.dims[0]}, &.{});
    const denom = try mean.add(eps_b);
    const inv = try denom.rsqrt();
    const inv_b = try inv.broadcast_in_dim(x.tensor.shape.dims, &.{0});
    return x.mul(inv_b);
}
