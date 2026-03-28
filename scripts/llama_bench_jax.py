#!/usr/bin/env python3
"""LLaMA training benchmark using jax-llm-examples (transparent pure-JAX model).

Comparison benchmark for Zigrad's LLaMA fine-tuning demo. Loads weights directly
from safetensors (no opaque HuggingFace model wrapper), runs forward+backward+SGD,
and outputs JSONL timing data.

The model implementation is the pure-JAX LLaMA 3 from the JAX team's reference repo
(reference/jax-llm-examples/llama3/). This gives us a transparent forward pass for
fair comparison with Zigrad's StableHLO-lowered equivalent.

Requirements:
    jax (with GPU support), safetensors, etils
    Reference model: reference/jax-llm-examples/ (cloned from jax-ml/jax-llm-examples)

Usage:
    task python -- scripts/llama_bench_jax.py --train --steps 20
    task python -- scripts/llama_bench_jax.py --train --steps 20 --seq 128 --batch 2
    task python -- scripts/llama_bench_jax.py --steps 10 --dtype f32
"""

import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as P

# Add jax-llm-examples to import path.
_REF_DIR = Path(__file__).resolve().parent.parent / "reference" / "jax-llm-examples" / "llama3"
if not _REF_DIR.is_dir():
    raise SystemExit(
        f"Reference model not found at {_REF_DIR}.\n"
        "Clone it: git clone https://github.com/jax-ml/jax-llm-examples.git reference/jax-llm-examples"
    )
sys.path.insert(0, str(_REF_DIR))

from llama3_jax import model as l3jax  # noqa: E402

# Mesh context manager (compatible across JAX versions).
from jax.sharding import set_mesh  # noqa: E402

try:
    from jax.sharding import use_mesh as set_mesh  # noqa: E402, F811
except ImportError:
    pass

try:
    from jax.sharding import AxisType
except ImportError:
    AxisType = None


def log_jsonl(event: str, **kwargs):
    """Write a JSONL record to stdout."""
    record = {"event": event, "time_ms": round(time.time() * 1000, 1)}
    record.update(kwargs)
    print(json.dumps(record), flush=True)


def eprint(*args, **kwargs):
    """Print to stderr."""
    print(*args, file=sys.stderr, **kwargs)


def load_weights(model_dir: Path, cfg: l3jax.Config, dtype) -> l3jax.Weights:
    """Load safetensors weights and convert to jax-llm-examples Weights format.

    Handles weight tying (lm_head absent in safetensors for 3.2-1B) and transposes
    from HuggingFace (out_features, in_features) layout to jax-llm-examples layout.
    """
    from safetensors import safe_open

    model_dir = Path(model_dir)
    st_files = sorted(model_dir.glob("*.safetensors"))
    if not st_files:
        raise FileNotFoundError(f"No safetensors files in {model_dir}")

    # Load all tensors as jax arrays (handles bf16 natively).
    tensors = {}
    for f in st_files:
        with safe_open(str(f), framework="flax") as sf:
            for key in sf.keys():
                tensors[key] = sf.get_tensor(key)

    cast = lambda arr: jnp.asarray(arr).astype(dtype)

    layers = []
    for i in range(cfg.num_layers):
        p = f"model.layers.{i}."
        # Weight shapes from HuggingFace (PyTorch nn.Linear stores (out, in)):
        #   q_proj: (q_heads*head_dim, embed) -> T -> reshape -> (embed, q_heads, head_dim)
        #   k/v_proj: (kv_heads*head_dim, embed) -> T -> reshape -> (embed, kv_heads, head_dim)
        #   o_proj: (embed, q_heads*head_dim) -> T -> reshape -> (q_heads, head_dim, embed)
        #   gate/up_proj: (ffw_size, embed) -> T -> (embed, ffw_size)
        #   down_proj: (embed, ffw_size) -> T -> (ffw_size, embed)
        layers.append(l3jax.Layer(
            q=cast(tensors[f"{p}self_attn.q_proj.weight"].T.reshape(cfg.embed, cfg.q_heads, cfg.head_dim)),
            k=cast(tensors[f"{p}self_attn.k_proj.weight"].T.reshape(cfg.embed, cfg.kv_heads, cfg.head_dim)),
            v=cast(tensors[f"{p}self_attn.v_proj.weight"].T.reshape(cfg.embed, cfg.kv_heads, cfg.head_dim)),
            o=cast(tensors[f"{p}self_attn.o_proj.weight"].T.reshape(cfg.q_heads, cfg.head_dim, cfg.embed)),
            w_gate=cast(tensors[f"{p}mlp.gate_proj.weight"].T),
            w_up=cast(tensors[f"{p}mlp.up_proj.weight"].T),
            w_down=cast(tensors[f"{p}mlp.down_proj.weight"].T),
            attn_pre_gamma=cast(tensors[f"{p}input_layernorm.weight"]),
            attn_post_gamma=cast(tensors[f"{p}post_attention_layernorm.weight"]),
        ))

    embedding = cast(tensors["model.embed_tokens.weight"])
    gamma_final = cast(tensors["model.norm.weight"])

    # Handle tied weights (llama-3.2-1b-instruct has tie_word_embeddings=true,
    # so lm_head.weight is absent from safetensors). Both Zigrad and this script
    # break the tie during training (separate gradient updates).
    if "lm_head.weight" in tensors:
        lm_head = cast(tensors["lm_head.weight"].T)
    else:
        lm_head = cast(tensors["model.embed_tokens.weight"]).T

    return l3jax.Weights(layers=layers, embedding=embedding, gamma_final=gamma_final, lm_head=lm_head)


def make_batch(seq: int, batch_size: int, active_len: int, vocab_size: int):
    """Create input batch matching Zigrad's llama_demo token pattern.

    Returns (tokens, target_ids, segment_ids, mask) as jax arrays.
    segment_ids serve as both attention mask (for the model) and loss mask.
    """
    # Same seed tokens as Zigrad's llama_demo.zig
    base_tokens = np.array([128000, 128009, 128001, 128008], dtype=np.int32)
    base_targets = np.array([128009, 128001, 128008, 128001], dtype=np.int32)

    tokens_1d = np.array([base_tokens[i % len(base_tokens)] % vocab_size for i in range(seq)], dtype=np.int32)
    targets_1d = np.array([base_targets[i % len(base_targets)] % vocab_size for i in range(seq)], dtype=np.int32)

    tokens = np.tile(tokens_1d[None, :], (batch_size, 1))
    target_ids = np.tile(targets_1d[None, :], (batch_size, 1))

    # segment_ids: 1 for active tokens, 0 for padding.
    # jax-llm-examples forward() uses segment_ids for causal mask construction.
    segment_ids_1d = np.zeros(seq, dtype=np.int32)
    segment_ids_1d[:active_len] = 1
    segment_ids = np.tile(segment_ids_1d[None, :], (batch_size, 1))

    # Float mask for loss computation (separate from segment_ids to avoid dtype issues).
    mask_1d = np.zeros(seq, dtype=np.float32)
    mask_1d[:active_len] = 1.0
    mask = np.tile(mask_1d[None, :], (batch_size, 1))

    return jnp.array(tokens), jnp.array(target_ids), jnp.array(segment_ids), jnp.array(mask)


def main():
    parser = argparse.ArgumentParser(description="LLaMA training benchmark (jax-llm-examples)")
    parser.add_argument("--warmup", type=int, default=5, help="warmup steps (excluded from timing)")
    parser.add_argument("--steps", type=int, default=20, help="timed steps")
    parser.add_argument("--train", action="store_true", help="run train step (forward+backward+SGD)")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "f32"], help="model dtype")
    parser.add_argument("--seq", type=int, default=4, help="sequence length")
    parser.add_argument("--batch", type=int, default=1, help="batch size")
    parser.add_argument("--active-len", type=int, default=None, help="active tokens per sequence (default: --seq)")
    parser.add_argument("--model-dir", default=None, help="model directory with config.json and *.safetensors")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate for SGD")
    parser.add_argument("--quiet", action="store_true", help="suppress stderr progress")
    args = parser.parse_args()

    model_dir = Path(args.model_dir or os.environ.get(
        "ZG_LLAMA_SAFETENSORS_PATH", "./weights/llama-3.2-1b-instruct"
    ))
    if model_dir.is_file():
        model_dir = model_dir.parent
    if not model_dir.is_dir():
        raise SystemExit(f"Model directory not found: {model_dir}")

    dtype = jnp.bfloat16 if args.dtype == "bf16" else jnp.float32
    seq = args.seq
    batch_size = args.batch
    active_len = args.active_len if args.active_len is not None else seq
    lr = args.lr

    if active_len > seq:
        raise SystemExit("--active-len must be <= --seq")

    # Load model config.
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise SystemExit(f"config.json not found in {model_dir}")
    cfg = l3jax.llama_to_jax_config(json.loads(config_path.read_text()))

    # Single-device mesh with Auto axis types and all sharding rules set to None.
    # The jax-llm-examples model uses out_sharding on .get() and einsum with axis
    # names from ShardingRules. Setting all rules to None produces P(None, None, ...)
    # which works with Auto axes and is VJP-compatible (Explicit axes break VJP of
    # bare reductions like rms_norm's jnp.mean).
    mesh_kwargs = {}
    if AxisType is not None:
        mesh_kwargs["axis_types"] = (AxisType.Auto,) * 3
    mesh = jax.make_mesh((1, 1, 1), ("x", "y", "z"), **mesh_kwargs)
    no_shard_rules = l3jax.ShardingRules(
        **{f.name: None for f in dataclasses.fields(l3jax.ShardingRules)}
    )
    cfg = dataclasses.replace(
        cfg,
        mesh=mesh,
        max_seq_len=max(seq, 128),
        dtype=dtype,
        quant_layer=False,
        quant_cache=False,
        use_prefill_attn_kernel=False,
        use_decode_attn_kernel=False,
        rules=no_shard_rules,
    )

    log_jsonl(
        "config",
        model=model_dir.name,
        model_dir=str(model_dir),
        framework="jax",
        model_impl="jax-llm-examples",
        backend=str(jax.default_backend()),
        dtype=args.dtype,
        seq=seq,
        batch=batch_size,
        active_len=active_len,
        train=args.train,
        lr=lr,
        warmup=args.warmup,
        steps=args.steps,
        num_layers=cfg.num_layers,
        embed=cfg.embed,
        q_heads=cfg.q_heads,
        kv_heads=cfg.kv_heads,
        head_dim=cfg.head_dim,
        ffw_size=cfg.ffw_size,
        vocab_size=cfg.vocab_size,
        jax_version=jax.__version__,
    )

    # Load weights on CPU to avoid GPU OOM during conversion.
    if not args.quiet:
        eprint("Loading weights...")
    cpu = jax.devices("cpu")[0]
    with jax.default_device(cpu):
        weights = load_weights(model_dir, cfg, dtype)

    device = jax.devices()[0]
    if not args.quiet:
        eprint(f"Transferring weights to {device}...")
    weights = jax.device_put(weights, device)

    # Build input batch.
    tokens, target_ids, segment_ids, mask = make_batch(seq, batch_size, active_len, cfg.vocab_size)

    # Loss function: cross-entropy matching Zigrad's manual logsumexp approach.
    # JAX's log_softmax is numerically equivalent (max-subtract + log-sum-exp internally).
    def loss_fn(weights, tokens, segment_ids, target_ids, mask):
        logits, _ = l3jax.forward(tokens, segment_ids, weights, cfg, cache=None)
        log_probs = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
        gathered = jnp.take_along_axis(log_probs, target_ids[..., None], axis=-1)[..., 0]
        return -jnp.sum(gathered * mask.astype(jnp.float32))

    if args.train:
        def step_fn(weights, tokens, segment_ids, target_ids, mask):
            loss, grads = jax.value_and_grad(loss_fn)(weights, tokens, segment_ids, target_ids, mask)
            new_weights = jax.tree.map(lambda w, g: w - lr * g, weights, grads)
            return loss, new_weights
    else:
        def step_fn(weights, tokens, segment_ids, target_ids, mask):
            return loss_fn(weights, tokens, segment_ids, target_ids, mask)

    step_fn = jax.jit(step_fn)

    with set_mesh(mesh):
        # Warmup (includes JIT compilation).
        if not args.quiet:
            eprint(f"Running {args.warmup} warmup steps...")
        for w in range(args.warmup):
            out = step_fn(weights, tokens, segment_ids, target_ids, mask)
            out = jax.block_until_ready(out)
            _ = float(out[0])
            if args.train:
                weights = out[1]
            if not args.quiet:
                eprint(f"  warmup {w + 1}/{args.warmup}")

        log_jsonl("warmup_end", warmup_steps=args.warmup)

        # Timed steps.
        if not args.quiet:
            eprint(f"Running {args.steps} timed steps...")

        total_ms = 0.0
        for step in range(args.steps):
            t0 = time.perf_counter()
            out = step_fn(weights, tokens, segment_ids, target_ids, mask)
            t01 = time.perf_counter()
            out = jax.block_until_ready(out)
            loss_value = float(out[0])
            t1 = time.perf_counter()

            if args.train:
                weights = out[1]

            dispatch_ms = (t01 - t0) * 1e3
            sync_read_ms = (t1 - t01) * 1e3
            step_ms = (t1 - t0) * 1e3
            total_ms += step_ms

            log_jsonl("step", step=step, loss=round(loss_value, 6), step_ms=round(step_ms, 3))
            if not args.quiet:
                mode = "train" if args.train else "eval"
                eprint(f"  step {step}: loss={loss_value:.6f} dispatch={dispatch_ms:.3f}ms sync+read={sync_read_ms:.3f}ms step={step_ms:.3f}ms [{mode}]")

    avg_ms = total_ms / max(args.steps, 1)
    log_jsonl(
        "summary",
        avg_step_ms=round(avg_ms, 3),
        steps=args.steps,
        warmup=args.warmup,
        seq=seq,
        batch=batch_size,
        train=args.train,
    )

    if not args.quiet:
        eprint(f"avg_step_ms={avg_ms:.3f} (warmup={args.warmup} steps={args.steps} seq={seq} batch={batch_size})")
        eprint("OK: llama-bench-jax executed")


if __name__ == "__main__":
    main()
