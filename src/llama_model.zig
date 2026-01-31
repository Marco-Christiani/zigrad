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
    q_proj: zg.frontend.Tensor,
    k_proj: zg.frontend.Tensor,
    v_proj: zg.frontend.Tensor,
    o_proj: zg.frontend.Tensor,
    gate_proj: zg.frontend.Tensor,
    up_proj: zg.frontend.Tensor,
    down_proj: zg.frontend.Tensor,
};

pub fn forward(
    tokens: zg.frontend.Tensor,
    mask: zg.frontend.Tensor,
    sin: zg.frontend.Tensor,
    cos: zg.frontend.Tensor,
    weights: LlamaWeights,
    eps: f32,
) !zg.frontend.Tensor {
    var x = try weights.w_emb.gather_rows(tokens);
    for (weights.layers) |layer| {
        x = try layer_forward(x, mask, sin, cos, layer, eps);
    }
    const final_norm = try rms_norm(x, weights.norm, eps);
    return final_norm.matmul(weights.w_out);
}

fn layer_forward(
    x_in: zg.frontend.Tensor,
    mask: zg.frontend.Tensor,
    sin: zg.frontend.Tensor,
    cos: zg.frontend.Tensor,
    layer: LayerWeights,
    eps: f32,
) !zg.frontend.Tensor {
    const x_norm = try rms_norm(x_in, layer.input_norm, eps);
    const attn_out = try self_attention(x_norm, mask, sin, cos, layer);
    const x1 = try x_in.add(attn_out);

    const post_norm = try rms_norm(x1, layer.post_norm, eps);
    const mlp_out = try mlp(post_norm, layer);
    return x1.add(mlp_out);
}

fn self_attention(
    x: zg.frontend.Tensor,
    mask: zg.frontend.Tensor,
    sin: zg.frontend.Tensor,
    cos: zg.frontend.Tensor,
    layer: LayerWeights,
) !zg.frontend.Tensor {
    const n_heads: usize = 32;
    const n_kv: usize = 8;
    const head_dim: usize = 64;
    const hidden: usize = n_heads * head_dim;
    const seq: usize = x.tensor.shape.dims[0];

    const q = try x.matmul(layer.q_proj);
    const k = try x.matmul(layer.k_proj);
    const v = try x.matmul(layer.v_proj);

    const q_3d = try q.reshape(&.{ seq, n_heads, head_dim });
    const k_3d = try k.reshape(&.{ seq, n_kv, head_dim });
    const v_3d = try v.reshape(&.{ seq, n_kv, head_dim });

    const q_rot = try apply_rope(q_3d, sin, cos);
    const k_rot = try apply_rope(k_3d, sin, cos);

    const k_rep = try repeat_kv(k_rot, n_heads / n_kv);
    const v_rep = try repeat_kv(v_3d, n_heads / n_kv);

    const q_t = try q_rot.transpose(&.{ 1, 0, 2 });
    const k_t = try k_rep.transpose(&.{ 1, 2, 0 });
    const v_t = try v_rep.transpose(&.{ 1, 0, 2 });

    const scores = try q_t.dot_general(k_t, .{
        .lhs_batch_dims = &.{0},
        .rhs_batch_dims = &.{0},
        .lhs_contracting_dims = &.{2},
        .rhs_contracting_dims = &.{1},
    });

    const scale = 1.0 / std.math.sqrt(@as(f32, @floatFromInt(head_dim)));
    const scale_lit = ops.types.scalar_literal(scores.tensor.dtype, scale);
    const scale_t = try scores.builder.scalar_literal(scale_lit);
    const scale_b = try scale_t.broadcast_in_dim(scores.tensor.shape.dims, &.{});
    const scaled = try scores.mul(scale_b);

    const masked = try apply_causal_mask(scaled, mask);
    const attn = try softmax_last_dim(masked);
    const out_heads = try attn.dot_general(v_t, .{
        .lhs_batch_dims = &.{0},
        .rhs_batch_dims = &.{0},
        .lhs_contracting_dims = &.{2},
        .rhs_contracting_dims = &.{1},
    });

    const out_t = try out_heads.transpose(&.{ 1, 0, 2 });
    const out = try out_t.reshape(&.{ seq, hidden });
    return out.matmul(layer.o_proj);
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

fn apply_causal_mask(scores: zg.frontend.Tensor, mask: zg.frontend.Tensor) !zg.frontend.Tensor {
    const zero = try mask.builder.scalar_literal(.{ .i32 = 0 });
    const zeros = try zero.broadcast_in_dim(mask.tensor.shape.dims, &.{});
    const cond = try mask.compare(zeros, .{ .direction = .GT, .compare_type = .SIGNED });
    const cond_b = try cond.broadcast_in_dim(scores.tensor.shape.dims, &.{ 1, 2 });

    const neg_lit = ops.types.scalar_literal(scores.tensor.dtype, -1.0e9);
    const neg = try scores.builder.scalar_literal(neg_lit);
    const neg_b = try neg.broadcast_in_dim(scores.tensor.shape.dims, &.{});
    return scores.select(cond_b, neg_b);
}

fn repeat_kv(x: zg.frontend.Tensor, repeat: usize) !zg.frontend.Tensor {
    const seq = x.tensor.shape.dims[0];
    const n_kv = x.tensor.shape.dims[1];
    const head_dim = x.tensor.shape.dims[2];
    const out = try x.broadcast_in_dim(&.{ seq, n_kv, repeat, head_dim }, &.{ 0, 1, 3 });
    return out.reshape(&.{ seq, n_kv * repeat, head_dim });
}

fn softmax_last_dim(x: zg.frontend.Tensor) !zg.frontend.Tensor {
    const rank = x.tensor.shape.dims.len;
    const axis: i64 = @intCast(rank - 1);
    const max = try x.reduce_max(&.{axis});

    var bd_buf: [3]i64 = .{ 0, 1, 2 };
    const bd = bd_buf[0 .. rank - 1];
    const max_b = try max.broadcast_in_dim(x.tensor.shape.dims, bd);

    const shifted = try x.sub(max_b);
    const exp = try shifted.exp();
    const sum = try exp.reduce_sum(&.{axis});
    const sum_b = try sum.broadcast_in_dim(x.tensor.shape.dims, bd);
    return exp.div(sum_b);
}

fn mlp(x: zg.frontend.Tensor, layer: LayerWeights) !zg.frontend.Tensor {
    const gate = try x.matmul(layer.gate_proj);
    const up = try x.matmul(layer.up_proj);
    const act = try gate.mul(try gate.logistic());
    const fused = try act.mul(up);
    return fused.matmul(layer.down_proj);
}

pub fn rms_norm(x: zg.frontend.Tensor, weight: zg.frontend.Tensor, eps: f32) !zg.frontend.Tensor {
    const hidden = x.tensor.shape.dims[1];
    const hidden_f: f32 = @floatFromInt(hidden);
    const mean_scale_lit = ops.types.scalar_literal(x.tensor.dtype, 1.0 / hidden_f);
    const mean_scale = try x.builder.scalar_literal(mean_scale_lit);

    const x_sq = try x.mul(x);
    const sum = try x_sq.reduce_sum(&.{1});
    const mean_scale_b = try mean_scale.broadcast_in_dim(&.{x.tensor.shape.dims[0]}, &.{});
    const mean = try sum.mul(mean_scale_b);

    const eps_lit = ops.types.scalar_literal(x.tensor.dtype, eps);
    const eps_tensor = try x.builder.scalar_literal(eps_lit);
    const eps_b = try eps_tensor.broadcast_in_dim(&.{x.tensor.shape.dims[0]}, &.{});
    const denom = try mean.add(eps_b);
    const inv = try denom.rsqrt();
    const inv_b = try inv.broadcast_in_dim(x.tensor.shape.dims, &.{0});
    const normed = try x.mul(inv_b);

    const w_b = try weight.broadcast_in_dim(x.tensor.shape.dims, &.{1});
    return normed.mul(w_b);
}
