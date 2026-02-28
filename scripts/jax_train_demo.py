import argparse
import time

import jax
import jax.numpy as jnp
from jax import tree_util


def loss_fn(params, batch):
    w1, b1, w2, b2, w3, b3 = params
    x, y = batch
    a1 = x @ w1 + b1
    a2 = a1 @ w2 + b2
    preds = a2 @ w3 + b3
    diff = preds - y
    return jnp.sum(diff * diff)


def dump_ir(loss_fn, params, batch, modes):
    if "jaxpr" in modes:
        jaxpr = jax.make_jaxpr(loss_fn)(params, batch)
        print(f"=== JAXPR {loss_fn.__name__} ===")
        print(jaxpr)
        print("")

    if "hlo" in modes or "stablehlo" in modes:
        lowered = jax.jit(loss_fn).lower(params, batch)
        if "hlo" in modes:
            print(f"=== HLO {loss_fn.__name__} ===")
            ir = lowered.compiler_ir("hlo")
            assert ir is not None
            ir_text = ir.as_hlo_text()
            print(ir_text)
            print("")
        if "stablehlo" in modes:
            print(f"=== StableHLO {loss_fn.__name__} ===")
            try:
                stable = lowered.compiler_ir("stablehlo")
                assert stable is not None
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


def main(args):
    ir_modes = {m.strip() for m in args.ir.split(",") if m.strip()}
    bs = 64
    in_dim = 784
    h1 = 128
    h2 = 64
    out_dim = 10

    def pattern(count, scale):
        idx = jnp.arange(count, dtype=jnp.float32) % 1024
        return idx * scale

    def init_w(shape, scale):
        return pattern(shape[0] * shape[1], scale).reshape(shape)

    def init_b(size, scale):
        return pattern(size, scale)

    true_w1 = init_w((in_dim, h1), 1e-6)
    true_b1 = init_b(h1, 1e-6)
    true_w2 = init_w((h1, h2), 1e-6)
    true_b2 = init_b(h2, 1e-6)
    true_w3 = init_w((h2, out_dim), 1e-6)
    true_b3 = init_b(out_dim, 1e-6)

    w1 = init_w((in_dim, h1), 1e-7)
    b1 = init_b(h1, 1e-7)
    w2 = init_w((h1, h2), 1e-7)
    b2 = init_b(h2, 1e-7)
    w3 = init_w((h2, out_dim), 1e-7)
    b3 = init_b(out_dim, 1e-7)

    x = (jnp.arange(bs * in_dim, dtype=jnp.float32) % 256).reshape(bs, in_dim) / 255.0
    y = x @ true_w1 + true_b1
    y = y @ true_w2 + true_b2
    y = y @ true_w3 + true_b3

    lr = 1e-2
    steps = args.steps

    params = (w1, b1, w2, b2, w3, b3)
    if ir_modes:
        dump_ir(loss_fn, params, (x, y), ir_modes)

    def step_fn(params, batch):
        loss, grads = jax.value_and_grad(loss_fn)(params, batch)
        w1, b1, w2, b2, w3, b3 = params
        gw1, gb1, gw2, gb2, gw3, gb3 = grads
        new_params = (
            w1 - lr * gw1,
            b1 - lr * gb1,
            w2 - lr * gw2,
            b2 - lr * gb2,
            w3 - lr * gw3,
            b3 - lr * gb3,
        )
        return loss, new_params

    step_fn = jax.jit(step_fn)

    if ir_modes:
        dump_ir(step_fn, params, (x, y), ir_modes)

    def block_all(loss, params):
        if args.block == "loss":
            return jax.block_until_ready(loss), params
        loss = jax.block_until_ready(loss)
        params = tree_util.tree_map(lambda x: x.block_until_ready(), params)
        return loss, params

    for _ in range(args.warmup):
        loss, params = step_fn(params, (x, y))
        loss, params = block_all(loss, params)

    total = 0.0
    total_compute = 0.0
    total_block = 0.0
    for step in range(steps):
        t0 = time.perf_counter()
        loss, params = step_fn(params, (x, y))
        t1 = time.perf_counter()
        loss, params = block_all(loss, params)
        t2 = time.perf_counter()
        compute_s = t1 - t0
        block_s = t2 - t1
        total_compute += compute_s
        total_block += block_s
        total += t2 - t0
        if not args.quiet:
            print(
                f"train-demo step {step}: loss={loss:.6f} compute_ms={compute_s * 1e3:.3f} block_ms={block_s * 1e3:.3f}"
            )

    avg_ms = (total / steps) * 1e3
    print(f"avg_step_ms={avg_ms:.3f}")
    print(f"avg_compute_ms={(total_compute / steps) * 1e3:.3f}")
    print(f"avg_block_ms={(total_block / steps) * 1e3:.3f}")


if __name__ == "__main__":
    # TODO: jaxtyping
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ir",
        default="",
        help="comma-separated list: jaxpr,hlo,stablehlo (empty disables dumps)",
    )
    parser.add_argument(
        "-w",
        "--warmup",
        type=int,
        default=1,
        help="number of warmup steps to exclude from timing",
    )
    parser.add_argument(
        "-s",
        "--steps",
        type=int,
        default=8,
        help="number of timed steps to run",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="suppress per-step prints (keeps only averages)",
    )
    parser.add_argument(
        "--block",
        default="all",
        choices=["loss", "all"],
        help="what to block on each step: loss or all outputs",
    )
    parser.add_argument(
        "-d",
        "--device",
        default="cpu",
        choices=["cpu", "gpu"],
    )
    args = parser.parse_args()
    device = jax.devices(args.device)[0]
    print("Using device", device)
    with jax.default_device(device):
        main(args)
