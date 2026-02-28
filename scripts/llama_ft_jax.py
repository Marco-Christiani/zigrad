import argparse
import contextlib
import importlib
import os
import time

import jax
import jax.numpy as jnp
from transformers import AutoConfig, FlaxAutoModelForCausalLM
from transformers.utils import logging as hf_logging


def loss_fn(model, params, batch, upcast: bool = False):
    input_ids, target_ids, attention_mask = batch
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        params=params,
        train=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    )
    logits = outputs.logits.astype(jnp.float32 if upcast else outputs.logits.dtype)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    gathered = jnp.take_along_axis(log_probs, target_ids[..., None], axis=-1)[..., 0]

    # match zigrad: mask loss by attention_mask (active tokens only)
    mask = attention_mask.astype(gathered.dtype)
    return -jnp.sum(gathered * mask)


def dump_ir(loss_fn_impl, model, params, batch, modes):
    if "jaxpr" in modes:
        jaxpr = jax.make_jaxpr(lambda p, b: loss_fn_impl(model, p, b))(params, batch)
        print("=== JAXPR ===")
        print(jaxpr)
        print("")

    if "hlo" in modes or "stablehlo" in modes:
        lowered = jax.jit(lambda p, b: loss_fn_impl(model, p, b)).lower(params, batch)
        if "hlo" in modes:
            print("=== HLO ===")
            ir = lowered.compiler_ir("hlo")
            if ir is None:
                return ValueError("Failed to lower to hlo, backend likely missing.")
            print(ir.as_hlo_text())
            print("")
        if "stablehlo" in modes:
            print("=== StableHLO ===")
            try:
                stable = lowered.compiler_ir("stablehlo")
                assert stable is not None, (
                    "Failed to lower to stablehlo, backend likely missing."
                )
                if hasattr(stable, "as_text"):
                    text = stable.as_text()
                elif hasattr(stable, "as_hlo_text"):
                    text = stable.as_hlo_text()
                else:
                    text = str(stable)
                print(text)
            except Exception as exc:
                print(f"(stablehlo unavailable: {exc})")
            print("")


def init_model(model_dir, dtype, max_pos: int):
    config = AutoConfig.from_pretrained(model_dir)
    config.use_cache = False
    # config.max_position_embeddings = 32768
    config.max_position_embeddings = max_pos

    # load on cpu to avoid gpu peak oom during init/conversion
    cpu = jax.devices("cpu")[0]
    with jax.default_device(cpu):
        print("Loading model")

        og_verb = hf_logging.get_verbosity()
        # suppress warnings about loading in a different dtype and uninitialized weights
        hf_logging.set_verbosity_error()
        model = FlaxAutoModelForCausalLM.from_pretrained(
            model_dir,
            config=config,
            dtype=dtype,
        )
        hf_logging.set_verbosity(og_verb)
        print("Casting params")
        params_cpu = model.params
        # tie lm_head.weight + embed_tokens.weight
        # pt might tie them automatically but not jax? not sure. either way, this addresses the uninitialized warning.
        params_cpu["lm_head"]["kernel"] = params_cpu["model"]["embed_tokens"][
            "embedding"
        ].T

    gpu_backend = os.environ.get("JAX_GPU_BACKEND", "gpu")
    print(f"Moving weights JAX_GPU_BACKEND={gpu_backend}")
    gpu = jax.devices(gpu_backend)[0]
    params = jax.device_put(params_cpu, gpu)
    return model, params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="warmup steps to exclude from timing",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help="timed steps to run",
    )
    parser.add_argument("--train", action="store_true", help="run train step")
    parser.add_argument(
        "--dtype",
        default="bf16",
        choices=["bf16", "f32"],
        help="model dtype",
    )
    parser.add_argument(
        "--seq",
        type=int,
        default=4,
        help="token sequence length",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="batch size (number of sequences)",
    )
    parser.add_argument(
        "--active-len",
        type=int,
        default=None,
        help="number of active (non-pad) tokens; defaults to --seq",
    )
    parser.add_argument(
        "--max-pos",
        type=int,
        default=129,
        help="max_position_embeddings for the model (must be >= --seq)",
    )
    parser.add_argument("--quiet", action="store_true", help="reduce output")
    parser.add_argument(
        "--ir",
        default="",
        help="comma-separated list: jaxpr,hlo,stablehlo (empty disables dumps)",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="model dir or safetensors path (defaults to ZG_LLAMA_SAFETENSORS_PATH or ./weights/llama-3.2-1b-instruct)",
    )
    args = parser.parse_args()

    model_path = args.model_path or os.environ.get(
        "ZG_LLAMA_SAFETENSORS_PATH", "./weights/llama-3.2-1b-instruct"
    )
    model_dir = (
        (os.path.dirname(model_path) or ".")
        if os.path.isfile(model_path)
        else model_path
    )

    if not os.path.exists(model_dir):
        raise SystemExit(
            "llama-ft-demo: model path not found; set --model-path or ZG_LLAMA_SAFETENSORS_PATH"
            f" (resolved {model_path=} {model_dir=})"
        )

    assert os.path.isdir(model_dir)

    dtype = jnp.bfloat16 if args.dtype == "bf16" else jnp.float32
    print(f"{dtype=}")
    if args.max_pos < args.seq:
        raise SystemExit("llama-ft-demo: --max-pos must be >= --seq")
    model, params = init_model(model_dir, dtype, args.max_pos)

    if not args.quiet:
        print(f"llama-ft-demo: loaded weights from {model_path}")

    seq = int(args.seq)
    batch_size = int(args.batch)
    if batch_size <= 0:
        raise SystemExit("llama-ft-demo: --batch must be >= 1")
    active_len = int(args.active_len) if args.active_len is not None else seq
    if active_len < 0 or active_len > seq:
        raise SystemExit("llama-ft-demo: --active-len must be in [0, --seq]")

    base_tokens = jnp.array([128000, 128009, 128001, 128008], dtype=jnp.int32)
    tokens_1d = jnp.take(
        base_tokens, jnp.arange(seq, dtype=jnp.int32) % base_tokens.shape[0]
    )
    tokens = jnp.tile(tokens_1d[None, :], (batch_size, 1))

    targets_1d = jnp.concatenate(
        [tokens_1d[1:], jnp.array([0], dtype=jnp.int32)], axis=0
    )
    targets = jnp.tile(targets_1d[None, :], (batch_size, 1))

    attention_mask_1d = (jnp.arange(seq, dtype=jnp.int32) < active_len).astype(
        jnp.int32
    )
    attention_mask = jnp.tile(attention_mask_1d[None, :], (batch_size, 1))
    batch = (tokens, targets, attention_mask)

    ir_modes = {m.strip() for m in args.ir.split(",") if m.strip()}
    if ir_modes:
        dump_ir(loss_fn, model, params, batch, ir_modes)

    lr = 1e-4

    if args.train:

        def step_fn_train(params, batch):
            loss, grads = jax.value_and_grad(loss_fn, argnums=1)(model, params, batch)
            new_params = jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads)
            return loss, new_params

        step_fn = step_fn_train
    else:

        def step_fn_eval(params, batch):
            loss = loss_fn(model, params, batch)
            return loss

        step_fn = step_fn_eval

    step_fn = jax.jit(step_fn)

    for _ in range(args.warmup):
        if args.train:
            loss, params = step_fn(params, batch)
            loss = jax.block_until_ready(loss)
        else:
            loss = step_fn(params, batch)
            loss = jax.block_until_ready(loss)

    total_ms = 0.0

    nvtx = None
    try:
        nvtx = importlib.import_module("nvtx")
    except Exception:
        nvtx = None

    if nvtx is not None:
        try:
            nvtx.range_push("llama-ft-demo timed loop")
        except Exception:
            pass

    for step in range(args.steps):
        t0 = time.perf_counter()
        out = step_fn(params, batch)
        t1 = time.perf_counter()

        # align with zigrad --quiet behavior: still block for device completion, but avoid host transfers/printing
        if args.quiet:
            t2 = time.perf_counter()
            out_ready = jax.block_until_ready(out)
            t3 = time.perf_counter()
            if args.train:
                loss, params = out_ready
            else:
                loss = out_ready
            dispatch_ms = (t1 - t0) * 1e3
            wait_ms = (t3 - t2) * 1e3
            loss_ms = 0.0
            total_step_ms = (t3 - t0) * 1e3
            loss_value = None
        else:
            if args.train:
                loss, params = out
            else:
                loss = out

            t2 = time.perf_counter()
            loss_ready = jax.block_until_ready(loss)
            t3 = time.perf_counter()
            loss_value = float(jax.device_get(loss_ready))
            t4 = time.perf_counter()

            dispatch_ms = (t1 - t0) * 1e3
            wait_ms = (t3 - t2) * 1e3
            loss_ms = (t4 - t3) * 1e3
            total_step_ms = (t4 - t0) * 1e3
        total_ms += total_step_ms

        if not args.quiet:
            print(
                f"llama-ft-demo step {step}: loss={loss_value:.6f} dispatch={dispatch_ms:.3f}ms "
                f"wait={wait_ms:.3f}ms loss={loss_ms:.3f}ms total={total_step_ms:.3f}ms"
            )

    avg_ms = total_ms / max(args.steps, 1)

    if nvtx is not None:
        with contextlib.suppress(Exception):
            nvtx.range_pop()

    print(
        f"llama-ft-demo avg_step_ms={avg_ms:.3f} (warmup={args.warmup} steps={args.steps} seq={seq} batch={batch_size})"
    )
    print("OK: llama-ft-demo executed")


if __name__ == "__main__":
    main()
