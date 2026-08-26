//! LLaMA 3.2 model expressed with Zigrad tensors.

const std = @import("std");
const zg = @import("zigrad");
const Tensor = zg.Tensor;

/// A checkpoint field containing one weight tensor.
///
/// The `weight` field preserves the final component of each SafeTensors key.
const Weight = struct {
    weight: Tensor,
};

/// Return LLaMA weights whose field paths match the checkpoint hierarchy.
///
/// Weights use PyTorch's `[out, in]` orientation. Linear projections express
///  that layout through their contracting dimensions.
///
/// The embedding matrix also supplies the output projection, so AD accumulates
///  both contributions into one parameter cotangent.
///
/// TODO(model): Represent checkpoints with an independent `lm_head.weight`.
pub fn LlamaWeights(comptime num_layers: usize) type {
    return struct {
        model: LlamaModel(num_layers),
    };
}

fn LlamaModel(comptime num_layers: usize) type {
    return struct {
        const Self = @This();

        embed_tokens: Weight,
        layers: [num_layers]LlamaLayer,
        norm: Weight,

        pub fn forward(
            self: Self,
            tokens: Tensor,
            mask: Tensor,
            attention_mask: Tensor,
            sin: Tensor,
            cos: Tensor,
            eps: f32,
        ) !Tensor {
            std.debug.assert(tokens.dims().len == 2);
            const batch_size = tokens.dims()[0];
            const seq = tokens.dims()[1];

            // build the shared predicates once
            const zero_mask = try Tensor.constant_like(mask, 0.0);
            const causal_pred = try mask.compare(zero_mask, .{ .direction = .GT, .compare_type = .FLOAT });
            const zero_attn_b = try Tensor.constant_like(attention_mask, 0.0);
            const attn_pred = try attention_mask.compare(zero_attn_b, .{ .direction = .GT, .compare_type = .FLOAT });

            var x = try self.embed(tokens);
            inline for (self.layers) |layer| {
                x = try layer.forward(x, causal_pred, attn_pred, sin, cos, eps);
            }
            return try self.output_head(x, batch_size, seq, eps);
        }

        /// Embedding lookup.
        ///
        /// Gathers `[B, S, hidden]` rows from `embed_tokens`.
        ///
        /// Accepts any integer token dtype, i32/i64 pass through and others
        ///  convert to i32.
        fn embed(self: Self, tokens: Tensor) !Tensor {
            const embed_tokens = self.embed_tokens.weight;
            std.debug.assert(embed_tokens.dims().len == 2);
            const batch_size = tokens.dims()[0];
            const seq = tokens.dims()[1];
            const hidden = embed_tokens.dims()[1];

            const token_ids = if (tokens.dtype == .i32 or tokens.dtype == .i64)
                tokens
            else
                try tokens.convert(.i32);
            const token_ids3 = try token_ids.reshape(&.{ batch_size, seq, 1 });
            return try embed_tokens.gather(token_ids3, .{
                .slice_sizes = &.{ 1, hidden },
                .offset_dims = &.{2},
                .collapsed_slice_dims = &.{0},
                .start_index_map = &.{0},
                .index_vector_dim = 2,
            });
        }

        /// Final RMS norm and output projection.
        ///
        /// The tied embedding matrix has `[vocab, hidden]` layout. Contracting
        ///  both hidden dimensions produces `[B*S, vocab]`.
        fn output_head(self: Self, x: Tensor, batch_size: i64, seq: i64, eps: f32) !Tensor {
            const embed_tokens = self.embed_tokens.weight;
            const vocab = embed_tokens.dims()[0];
            const hidden = embed_tokens.dims()[1];

            const final_norm = try rms_norm(x, self.norm.weight, eps);
            const final_norm_dot = try final_norm.convert(embed_tokens.dtype);
            const final_flat = try final_norm_dot.reshape(&.{ batch_size * seq, hidden });

            const logits_flat = try final_flat.dot_general(embed_tokens, .{
                .lhs_batch_dims = &.{},
                .rhs_batch_dims = &.{},
                .lhs_contracting_dims = &.{1},
                .rhs_contracting_dims = &.{1},
            });
            return try logits_flat.reshape(&.{ batch_size, seq, vocab });
        }
    };
}

const LlamaLayer = struct {
    self_attn: SelfAttention,
    mlp: MLP,
    input_layernorm: Weight,
    post_attention_layernorm: Weight,

    fn forward(
        self: LlamaLayer,
        x_in: Tensor,
        causal_pred: Tensor,
        attn_pred: Tensor,
        sin: Tensor,
        cos: Tensor,
        eps: f32,
    ) !Tensor {
        const x_norm = try rms_norm(x_in, self.input_layernorm.weight, eps);
        const attn_out = try self.self_attn.forward(x_norm, causal_pred, attn_pred, sin, cos);
        const x1 = try x_in.add(attn_out);

        const post_norm = try rms_norm(x1, self.post_attention_layernorm.weight, eps);
        const mlp_out = try self.mlp.forward(post_norm);
        return try x1.add(mlp_out);
    }
};

const SelfAttention = struct {
    q_proj: Weight,
    k_proj: Weight,
    v_proj: Weight,
    o_proj: Weight,

    fn forward(
        self: SelfAttention,
        x: Tensor,
        causal_pred: Tensor,
        attn_pred: Tensor,
        sin: Tensor,
        cos: Tensor,
    ) !Tensor {
        const q_proj = self.q_proj.weight;
        const k_proj = self.k_proj.weight;
        const v_proj = self.v_proj.weight;
        const o_proj = self.o_proj.weight;

        std.debug.assert(sin.dims().len == 2);
        std.debug.assert(cos.dims().len == 2);
        const rope_half = sin.dims()[1];
        std.debug.assert(cos.dims()[1] == rope_half);
        const head_dim = rope_half * 2;

        std.debug.assert(x.dims().len == 3);
        const batch_size = x.dims()[0];
        const seq = x.dims()[1];
        const hidden = x.dims()[2];

        // torch layout: q_proj is [n_heads*head_dim, hidden], k/v are [n_kv*head_dim, hidden]
        std.debug.assert(q_proj.dims().len == 2);
        std.debug.assert(q_proj.dims()[1] == hidden);
        const q_out = q_proj.dims()[0];
        std.debug.assert(@rem(q_out, head_dim) == 0);
        const n_heads = @divExact(q_out, head_dim);

        std.debug.assert(k_proj.dims().len == 2);
        std.debug.assert(v_proj.dims().len == 2);
        std.debug.assert(k_proj.dims()[1] == hidden);
        std.debug.assert(v_proj.dims()[1] == hidden);
        std.debug.assert(k_proj.dims()[0] == v_proj.dims()[0]);
        const kv_out = k_proj.dims()[0];
        std.debug.assert(@rem(kv_out, head_dim) == 0);
        const n_kv = @divExact(kv_out, head_dim);
        std.debug.assert(n_kv > 0);
        std.debug.assert(@rem(n_heads, n_kv) == 0);

        const x_dot = try x.convert(q_proj.dtype);
        const x_flat = try x_dot.reshape(&.{ batch_size * seq, hidden });

        const q_flat = try linear(x_flat, q_proj);
        const k_flat = try linear(x_flat, k_proj);
        const v_flat = try linear(x_flat, v_proj);

        const q = try q_flat.reshape(&.{ batch_size, seq, n_heads, head_dim });
        const k = try k_flat.reshape(&.{ batch_size, seq, n_kv, head_dim });
        const v = try v_flat.reshape(&.{ batch_size, seq, n_kv, head_dim });

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

        const scores_f32 = try scores.convert(.f32);

        const scale = 1.0 / std.math.sqrt(@as(f32, @floatFromInt(head_dim)));
        const scaled = try scores_f32.mul(try Tensor.constant_like(scores_f32, scale));

        const masked = try apply_attention_masks(scaled, causal_pred, attn_pred);

        // Keep softmax math in f32 even when the model dtype is bf16.
        //  bf16 exp/sum loses precision over long sequences. Matches jax
        //  default behavior, which is used for numerical reference.
        const attn_f32 = try softmax_last_dim_accum_f32(masked);
        const attn = try attn_f32.convert(v_rep.dtype);

        // out_bhsd: [B,H,S,D]
        // attn: [B,H,S,S], v_rep: [B,S,H,D]
        const out_bhsd = try attn.dot_general(v_rep, .{
            .lhs_batch_dims = &.{ 0, 1 },
            .rhs_batch_dims = &.{ 0, 2 },
            .lhs_contracting_dims = &.{3},
            .rhs_contracting_dims = &.{1},
        });

        const out_bshd = try out_bhsd.transpose(&.{ 0, 2, 1, 3 });

        const out_dot = try out_bshd.convert(o_proj.dtype);
        const out_flat = try out_dot.reshape(&.{ batch_size * seq, hidden });
        const proj_flat = try linear(out_flat, o_proj);
        return try proj_flat.reshape(&.{ batch_size, seq, hidden });
    }
};

const MLP = struct {
    gate_proj: Weight,
    up_proj: Weight,
    down_proj: Weight,

    fn forward(self: MLP, x: Tensor) !Tensor {
        const gate_proj = self.gate_proj.weight;
        const up_proj = self.up_proj.weight;
        const down_proj = self.down_proj.weight;

        if (x.dims().len == 3) {
            const batch_size = x.dims()[0];
            const seq = x.dims()[1];
            const hidden = x.dims()[2];
            const x_dot = try x.convert(gate_proj.dtype);
            const x_flat = try x_dot.reshape(&.{ batch_size * seq, hidden });
            const gate = try linear(x_flat, gate_proj);
            const up = try linear(x_flat, up_proj);
            const act = try silu_like(gate);
            const fused = try act.mul(up);
            const fused_dot = try fused.convert(down_proj.dtype);
            const down_flat = try linear(fused_dot, down_proj);
            return try down_flat.reshape(&.{ batch_size, seq, hidden });
        }

        std.debug.assert(x.dims().len == 2);
        const gate = try linear(x, gate_proj);
        const up = try linear(x, up_proj);
        const act = try silu_like(gate);
        const fused = try act.mul(up);
        return try linear(fused, down_proj);
    }
};

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

    return try y1.concatenate(&.{y2}, 3);
}

fn apply_attention_masks(scores: Tensor, causal_pred: Tensor, attn_pred: Tensor) !Tensor {
    const neg_inf: f64 = -std.math.inf(f64);
    const neg_b = try Tensor.constant_like(scores, neg_inf);

    if (scores.dims().len == 3) {
        // scores: [BH, S, S], causal_pred: [S,S], attn_pred: [BH,S]
        const causal_b = try causal_pred.broadcast_in_dim(scores.dims(), &.{ 1, 2 });
        const masked_causal = try scores.select(causal_b, neg_b);
        const attn_b = try attn_pred.broadcast_in_dim(scores.dims(), &.{ 0, 2 });
        return try masked_causal.select(attn_b, neg_b);
    }

    std.debug.assert(scores.dims().len == 4);
    // scores: [B,H,S,S], causal_pred: [S,S], attn_pred: [B,S]
    const causal_b = try causal_pred.broadcast_in_dim(scores.dims(), &.{ 2, 3 });
    const masked_causal = try scores.select(causal_b, neg_b);
    const attn_b = try attn_pred.broadcast_in_dim(scores.dims(), &.{ 0, 3 });
    return try masked_causal.select(attn_b, neg_b);
}

fn repeat_kv_bshd(x: Tensor, repeat: i64) !Tensor {
    std.debug.assert(x.dims().len == 4);
    const batch_size = x.dims()[0];
    const seq = x.dims()[1];
    const n_kv = x.dims()[2];
    const head_dim = x.dims()[3];
    const out = try x.broadcast_in_dim(&.{ batch_size, seq, n_kv, repeat, head_dim }, &.{ 0, 1, 2, 4 });
    return try out.reshape(&.{ batch_size, seq, n_kv * repeat, head_dim });
}

fn softmax_last_dim(x: Tensor) !Tensor {
    const rank = x.dims().len;
    const axis: i64 = @intCast(rank - 1);
    const max = try x.reduce(.{ .axes = &.{axis}, .operation = .maximum });

    std.debug.assert(rank <= 4);
    var bd_buf: [4]i64 = .{ 0, 1, 2, 3 };
    const bd = bd_buf[0 .. rank - 1];
    const max_b = try max.broadcast_in_dim(x.dims(), bd);

    const shifted = try x.sub(max_b);
    const exp = try shifted.exp();
    const sum = try exp.reduce(.{ .axes = &.{axis}, .operation = .sum });
    const sum_b = try sum.broadcast_in_dim(x.dims(), bd);
    return try exp.div(sum_b);
}

// TODO(api): Add a tensor softmax operation with an accumulation dtype.
fn softmax_last_dim_accum_f32(x: Tensor) !Tensor {
    // Only bf16 needs the f32 accumulation round-trip, everything else
    //  computes softmax in its native dtype.
    if (x.dtype != .bf16) return try softmax_last_dim(x);

    const x_f32 = try x.convert(.f32);
    const y_f32 = try softmax_last_dim(x_f32);
    return try y_f32.convert(.bf16);
}

/// Apply a linear projection whose weight uses `[out, in]` layout.
fn linear(x: Tensor, w: Tensor) !Tensor {
    return try x.dot_general(w, .{
        .lhs_batch_dims = &.{},
        .rhs_batch_dims = &.{},
        .lhs_contracting_dims = &.{1},
        .rhs_contracting_dims = &.{1},
    });
}

// TODO(api): Add a tensor SiLU operation with an accumulation dtype.
fn silu_like(x: Tensor) !Tensor {
    // silu(x) = x / (1 + exp(-x))
    // For bf16, compute exp in the model dtype and accumulate the rest in f32
    if (x.dtype != .bf16 and x.dtype != .f32) {
        // use the direct logistic lowering for other dtypes
        return try x.mul(try x.logistic());
    }

    const orig_dtype = x.dtype;
    const x_f32 = try x.convert(.f32);

    const zero_b = try Tensor.constant_like(x_f32, 0.0);
    const neg = try zero_b.sub(x_f32);

    // Round-trip through the model dtype before the f32 accumulation, no-op
    //  when `orig_dtype == .f32`.
    const exp = try (try neg.convert(orig_dtype)).exp();
    const exp_f32 = try exp.convert(.f32);

    const one_b = try Tensor.constant_like(x_f32, 1.0);
    const denom = try exp_f32.add(one_b);
    const inv = try one_b.div(denom);
    const y_f32 = try x_f32.mul(inv);

    return try y_f32.convert(orig_dtype);
}

fn rms_norm(x: Tensor, weight: Tensor, eps: f32) !Tensor {
    if (x.dims().len == 3) {
        const h = x.dims()[2];
        const orig_dtype = x.dtype;
        const x_f32 = try x.convert(.f32);
        const w_f32 = try weight.convert(.f32);

        const hidden_f: f32 = @floatFromInt(h);

        const x_sq = try x_f32.mul(x_f32);
        const sum = try x_sq.reduce(.{ .axes = &.{2}, .operation = .sum });
        const mean = try sum.mul(try Tensor.constant_like(sum, 1.0 / hidden_f));
        const denom = try mean.add(try Tensor.constant_like(mean, eps));
        const inv = try denom.rsqrt();
        const inv_b = try inv.broadcast_in_dim(x_f32.dims(), &.{ 0, 1 });
        const normed = try x_f32.mul(inv_b);

        const w_b = try w_f32.broadcast_in_dim(x_f32.dims(), &.{2});
        const y_f32 = try normed.mul(w_b);
        return try y_f32.convert(orig_dtype);
    }

    std.debug.assert(x.dims().len == 2);
    const orig_dtype = x.dtype;
    const x_f32 = try x.convert(.f32);
    const w_f32 = try weight.convert(.f32);

    const normed = try rms_norm_no_weight_f32(x_f32, eps);
    const w_b = try w_f32.broadcast_in_dim(normed.dims(), &.{1});
    const y_f32 = try normed.mul(w_b);
    return try y_f32.convert(orig_dtype);
}

fn rms_norm_no_weight_f32(x: Tensor, eps: f32) !Tensor {
    std.debug.assert(x.dtype == .f32);

    const hidden = x.dims()[1];
    const hidden_f: f32 = @floatFromInt(hidden);

    const x_sq = try x.mul(x);
    const sum = try x_sq.reduce(.{ .axes = &.{1}, .operation = .sum });
    const mean = try sum.mul(try Tensor.constant_like(sum, 1.0 / hidden_f));
    const denom = try mean.add(try Tensor.constant_like(mean, eps));
    const inv = try denom.rsqrt();
    const inv_b = try inv.broadcast_in_dim(x.dims(), &.{0});
    return try x.mul(inv_b);
}
