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

pub const CanonicalizeConfig = struct {
    /// Use an explicit batch dimension of 1 for attention QKV projection.
    qkv_batch1: bool = false,
    /// Use an explicit batch dimension of 1 for attention output projection.
    o_proj_batch1: bool = false,
    /// Use an explicit batch dimension of 1 for MLP GEMMs.
    mlp_batch1: bool = false,
};

pub const ForwardConfig = struct {
    canonicalize: CanonicalizeConfig = .{},
};

pub fn forward(
    tokens: zg.frontend.Tensor,
    mask: zg.frontend.Tensor,
    attention_mask: zg.frontend.Tensor,
    sin: zg.frontend.Tensor,
    cos: zg.frontend.Tensor,
    weights: LlamaWeights,
    cfg: ForwardConfig,
    eps: f32,
) !zg.frontend.Tensor {
    std.debug.assert(tokens.tensor.shape.dims.len == 2);
    const batch_size: usize = tokens.tensor.shape.dims[0];
    const seq: usize = tokens.tensor.shape.dims[1];
    const hidden: usize = 2048;

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

    const n: usize = batch_size * seq;
    var dims_n: [1]usize = .{n};
    const flat_tokens = try tokens.reshape(dims_n[0..]);
    const flat = try weights.w_emb.gather_rows(flat_tokens);
    const x0 = try flat.reshape(&.{ batch_size, seq, hidden });

    var x = x0;
    for (weights.layers) |layer| {
        x = try layer_forward(x, causal_pred, attn_pred, sin, cos, layer, &cfg, eps);
    }
    const final_norm = try rms_norm(x, weights.norm, eps);
    const flat_norm = try final_norm.reshape(&.{ n, hidden });
    const logits2 = try flat_norm.matmul(weights.w_out);
    return logits2.reshape(&.{ batch_size, seq, 128256 });
}

fn layer_forward(
    x_in: zg.frontend.Tensor,
    causal_pred: zg.frontend.Tensor,
    attn_pred: zg.frontend.Tensor,
    sin: zg.frontend.Tensor,
    cos: zg.frontend.Tensor,
    layer: LayerWeights,
    cfg: *const ForwardConfig,
    eps: f32,
) !zg.frontend.Tensor {
    const x_norm = try rms_norm(x_in, layer.input_norm, eps);
    const attn_out = try self_attention(x_norm, causal_pred, attn_pred, sin, cos, layer, cfg);
    const x1 = try x_in.add(attn_out);

    const post_norm = try rms_norm(x1, layer.post_norm, eps);
    const mlp_out = try mlp(post_norm, layer, cfg);
    return x1.add(mlp_out);
}

fn self_attention(
    x: zg.frontend.Tensor,
    causal_pred: zg.frontend.Tensor,
    attn_pred: zg.frontend.Tensor,
    sin: zg.frontend.Tensor,
    cos: zg.frontend.Tensor,
    layer: LayerWeights,
    cfg: *const ForwardConfig,
) !zg.frontend.Tensor {
    const n_heads: usize = 32;
    const n_kv: usize = 8;
    const head_dim: usize = 64;
    const hidden: usize = n_heads * head_dim;
    std.debug.assert(x.tensor.shape.dims.len == 3);
    const batch_size: usize = x.tensor.shape.dims[0];
    const seq: usize = x.tensor.shape.dims[1];
    const n: usize = batch_size * seq;

    const qkv = if (cfg.canonicalize.qkv_batch1) blk: {
        const x2 = try x.reshape(&.{ n, hidden });
        const qkv2 = try matmul_batch1(x2, layer.qkv_proj);
        break :blk try qkv2.reshape(&.{ batch_size, seq, hidden * 3 });
    } else blk: {
        const x_dot = if (x.tensor.dtype == layer.qkv_proj.tensor.dtype)
            x
        else
            try x.convert(layer.qkv_proj.tensor.dtype);
        var lhs_contract: [1]i64 = .{2};
        var rhs_contract: [1]i64 = .{0};
        break :blk try x_dot.dot_general(layer.qkv_proj, .{
            .lhs_batch_dims = &.{},
            .rhs_batch_dims = &.{},
            .lhs_contracting_dims = lhs_contract[0..],
            .rhs_contracting_dims = rhs_contract[0..],
        });
    };
    const qkv_n: i64 = @intCast(layer.qkv_proj.tensor.shape.dims[1]);
    const q2 = try qkv.slice(&.{ 0, 0, 0 }, &.{ @intCast(batch_size), @intCast(seq), 2048 }, &.{ 1, 1, 1 });
    const k2 = try qkv.slice(&.{ 0, 0, 2048 }, &.{ @intCast(batch_size), @intCast(seq), 2560 }, &.{ 1, 1, 1 });
    const v2 = try qkv.slice(&.{ 0, 0, 2560 }, &.{ @intCast(batch_size), @intCast(seq), qkv_n }, &.{ 1, 1, 1 });

    const q = try q2.reshape(&.{ batch_size, seq, n_heads, head_dim });
    const k = try k2.reshape(&.{ batch_size, seq, n_kv, head_dim });
    const v = try v2.reshape(&.{ batch_size, seq, n_kv, head_dim });

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
    const out = try out_bshd.reshape(&.{ batch_size, seq, hidden });
    if (cfg.canonicalize.o_proj_batch1) {
        const out2 = try out.reshape(&.{ n, hidden });
        const proj2 = try matmul_batch1(out2, layer.o_proj);
        return proj2.reshape(&.{ batch_size, seq, hidden });
    }

    const out_dot = if (out.tensor.dtype == layer.o_proj.tensor.dtype)
        out
    else
        try out.convert(layer.o_proj.tensor.dtype);
    var out_contract: [1]i64 = .{2};
    var w_contract: [1]i64 = .{0};
    return out_dot.dot_general(layer.o_proj, .{
        .lhs_batch_dims = &.{},
        .rhs_batch_dims = &.{},
        .lhs_contracting_dims = out_contract[0..],
        .rhs_contracting_dims = w_contract[0..],
    });
}

fn apply_rope(x: zg.frontend.Tensor, sin: zg.frontend.Tensor, cos: zg.frontend.Tensor) !zg.frontend.Tensor {
    const seq = x.tensor.shape.dims[0];
    const heads = x.tensor.shape.dims[1];
    const head_dim = x.tensor.shape.dims[2];
    const half = head_dim / 2;

    const x4 = try x.reshape(&.{ seq, heads, half, 2 });
    const seq_i: i64 = @intCast(seq);
    const heads_i: i64 = @intCast(heads);
    const half_i: i64 = @intCast(half);
    const even = try x4.slice(&.{ 0, 0, 0, 0 }, &.{ seq_i, heads_i, half_i, 1 }, &.{ 1, 1, 1, 1 });
    const odd = try x4.slice(&.{ 0, 0, 0, 1 }, &.{ seq_i, heads_i, half_i, 2 }, &.{ 1, 1, 1, 1 });

    const even_3 = try even.reshape(&.{ seq, heads, half });
    const odd_3 = try odd.reshape(&.{ seq, heads, half });

    const sin_b = try sin.broadcast_in_dim(&.{ seq, heads, half }, &.{ 0, 2 });
    const cos_b = try cos.broadcast_in_dim(&.{ seq, heads, half }, &.{ 0, 2 });

    const even_cos = try even_3.mul(cos_b);
    const odd_sin = try odd_3.mul(sin_b);
    const out_even = try even_cos.sub(odd_sin);

    const even_sin = try even_3.mul(sin_b);
    const odd_cos = try odd_3.mul(cos_b);
    const out_odd = try even_sin.add(odd_cos);

    const out_even4 = try out_even.reshape(&.{ seq, heads, half, 1 });
    const out_odd4 = try out_odd.reshape(&.{ seq, heads, half, 1 });
    const interleaved = try out_even4.concatenate(&.{out_odd4}, 3);
    return interleaved.reshape(&.{ seq, heads, head_dim });
}

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
    const hd_i64: i64 = @intCast(head_dim);

    const x1 = try x.slice(&.{ 0, 0, 0, 0 }, &.{ b_i64, s_i64, h_i64, half_i64 }, &.{ 1, 1, 1, 1 });
    const x2 = try x.slice(&.{ 0, 0, 0, half_i64 }, &.{ b_i64, s_i64, h_i64, hd_i64 }, &.{ 1, 1, 1, 1 });

    const sin_b = try sin.broadcast_in_dim(&.{ batch_size, seq, heads, half }, &.{ 1, 3 });
    const cos_b = try cos.broadcast_in_dim(&.{ batch_size, seq, heads, half }, &.{ 1, 3 });

    const a = try x1.mul(cos_b);
    const b = try x2.mul(sin_b);
    const y1 = try a.sub(b);

    const c = try x1.mul(sin_b);
    const d = try x2.mul(cos_b);
    const y2 = try c.add(d);

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

fn mlp(x: zg.frontend.Tensor, layer: LayerWeights, cfg: *const ForwardConfig) !zg.frontend.Tensor {
    if (x.tensor.shape.dims.len == 3) {
        const b = x.tensor.shape.dims[0];
        const s = x.tensor.shape.dims[1];
        const h = x.tensor.shape.dims[2];
        if (cfg.canonicalize.mlp_batch1) {
            const n = b * s;
            const x2 = try x.reshape(&.{ n, h });
            const y2 = try mlp(x2, layer, cfg);
            return y2.reshape(&.{ b, s, h });
        }

        var lhs_contract: [1]i64 = .{2};
        var rhs_contract: [1]i64 = .{0};
        const x_dot = if (x.tensor.dtype == layer.gate_proj.tensor.dtype)
            x
        else
            try x.convert(layer.gate_proj.tensor.dtype);
        const gate = try x_dot.dot_general(layer.gate_proj, .{
            .lhs_batch_dims = &.{},
            .rhs_batch_dims = &.{},
            .lhs_contracting_dims = lhs_contract[0..],
            .rhs_contracting_dims = rhs_contract[0..],
        });
        const up = try x_dot.dot_general(layer.up_proj, .{
            .lhs_batch_dims = &.{},
            .rhs_batch_dims = &.{},
            .lhs_contracting_dims = lhs_contract[0..],
            .rhs_contracting_dims = rhs_contract[0..],
        });
        const act = try silu_like(gate);
        const fused = try act.mul(up);
        const fused_dot = if (fused.tensor.dtype == layer.down_proj.tensor.dtype)
            fused
        else
            try fused.convert(layer.down_proj.tensor.dtype);
        return fused_dot.dot_general(layer.down_proj, .{
            .lhs_batch_dims = &.{},
            .rhs_batch_dims = &.{},
            .lhs_contracting_dims = lhs_contract[0..],
            .rhs_contracting_dims = rhs_contract[0..],
        });
    }

    std.debug.assert(x.tensor.shape.dims.len == 2);
    const gate = if (cfg.canonicalize.mlp_batch1) try matmul_batch1(x, layer.gate_proj) else try x.matmul(layer.gate_proj);
    const up = if (cfg.canonicalize.mlp_batch1) try matmul_batch1(x, layer.up_proj) else try x.matmul(layer.up_proj);
    const act = try silu_like(gate);
    const fused = try act.mul(up);
    return if (cfg.canonicalize.mlp_batch1)
        matmul_batch1(fused, layer.down_proj)
    else
        fused.matmul(layer.down_proj);
}

fn matmul_batch1(x: zg.frontend.Tensor, w: zg.frontend.Tensor) !zg.frontend.Tensor {
    std.debug.assert(x.tensor.shape.dims.len == 2);
    std.debug.assert(w.tensor.shape.dims.len == 2);
    std.debug.assert(x.tensor.shape.dims[1] == w.tensor.shape.dims[0]);

    const seq: usize = x.tensor.shape.dims[0];
    const k: usize = x.tensor.shape.dims[1];
    const n: usize = w.tensor.shape.dims[1];

    // Express (M,K)x(K,N)->(M,N) as a batched dot_general with batch=1.
    // This matches the dot_general VJP pattern we already support (batch dims on both operands).
    const x3 = try x.reshape(&.{ 1, seq, k });
    const w3 = try w.reshape(&.{ 1, k, n });

    var batch: [1]i64 = .{0};
    var lhs_contract: [1]i64 = .{2};
    var rhs_contract: [1]i64 = .{1};
    const y3 = try x3.dot_general(w3, .{
        .lhs_batch_dims = batch[0..],
        .rhs_batch_dims = batch[0..],
        .lhs_contracting_dims = lhs_contract[0..],
        .rhs_contracting_dims = rhs_contract[0..],
    });
    const y2 = try y3.reshape(&.{ seq, n });
    return y2;
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
