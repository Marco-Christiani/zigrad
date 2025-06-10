# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy",
#     "jax",
#     "flax",
#     "optax",
# ]
# ///
import os
import platform
import time
from pathlib import Path

import numpy as np
import jax
from jax import jit
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax


class Model(nn.Module):
    variant: str = "simple"

    @nn.compact
    def __call__(self, x):
        if self.variant == "simple":
            dim_fc1, dim_fc2, dim_fc3 = 128, 64, 10
        elif self.variant == "simple2":
            dim_fc1, dim_fc2, dim_fc3 = 784, 128, 10
        else:
            raise ValueError("Invalid variant")

        x = x.reshape(x.shape[0], -1)  # Flatten
        x = nn.Dense(dim_fc1, kernel_init=nn.initializers.kaiming_normal())(x)
        x = nn.relu(x)
        x = nn.Dense(dim_fc2, kernel_init=nn.initializers.kaiming_normal())(x)
        x = nn.relu(x)
        x = nn.Dense(dim_fc3, kernel_init=nn.initializers.kaiming_normal())(x)
        return x


def load_mnist_all_at_once(filepath: Path, scale: bool):
    """Load all data into memory at once as JAX arrays"""
    print(f"Loading data from {filepath}")
    data = np.loadtxt(filepath, delimiter=",")
    print(f"Data shape: {data.shape}")

    if data.shape[1] != 794:
        raise ValueError(f"Unexpected data shape. Expected 794 columns, got {data.shape[1]}")

    images = data[:, 10:].astype(np.float32)
    if scale and images.max() > 1:
        print("Scaling pixel values")
        images /= 255.0

    labels = data[:, :10].astype(np.float32)
    images = images.reshape(-1, 28, 28, 1)

    return jnp.array(images), jnp.array(labels)


def get_batch(images, labels, idx, batch_size):
    start_idx = idx * batch_size
    end_idx = start_idx + batch_size
    return images[start_idx:end_idx], labels[start_idx:end_idx]


class Profiler:
    def __init__(self, total_batches):
        self.epoch = 0
        self.batch = 0
        self.total_time = 0
        self.total_batches = total_batches
        self.losses = []  # Track all losses for statistics

    def log(self, loss, duration_ms):
        print(f"train_loss: {loss:.5f} [{self.batch}/{self.total_batches}] [ms/sample: {duration_ms:.4f}]")
        self.losses.append(loss)

    def log_epoch(self, avg_loss, duration_ms):
        print(f"Epoch {self.epoch + 1}: Avg Loss = {avg_loss:.4f} ({duration_ms:.2f}ms)")

    def log_final_stats(self):
        losses = np.array(self.losses)
        print(f"Loss s={np.std(losses):.5} mu={np.mean(losses):.5}")


def create_train_state(rng, learning_rate, model, input_shape):
    params = model.init(rng, jnp.ones(input_shape))
    tx = optax.sgd(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


@jit
def cross_entropy_loss(logits, labels):
    return -jnp.mean(jnp.sum(labels * jax.nn.log_softmax(logits), axis=-1))


@jit
def train_step(state, batch_images, batch_labels):
    def loss_fn(params):
        logits = state.apply_fn(params, batch_images)
        loss = cross_entropy_loss(logits, batch_labels)
        return loss, logits

    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss, logits


@jit
def eval_step(state, batch_images, batch_labels):
    logits = state.apply_fn(state.params, batch_images)
    loss = cross_entropy_loss(logits, batch_labels)
    return loss, logits


def train_epoch(state, images, labels, batch_size, profiler):
    num_batches = len(images) // batch_size
    epoch_loss = []
    epoch_start_time = time.perf_counter()

    for i in range(num_batches):
        profiler.batch = i + 1
        batch_start_time = time.perf_counter()

        batch_images, batch_labels = get_batch(images, labels, i, batch_size)
        state, loss, _ = train_step(state, batch_images, batch_labels)

        # Force sync for accurate timing
        loss = loss.block_until_ready()
        epoch_loss.append(loss.item())

        batch_end_time = time.perf_counter()
        duration_ms = (batch_end_time - batch_start_time) * 1000 / batch_size
        profiler.log(loss.item(), duration_ms)

    avg_loss = np.mean(epoch_loss)
    epoch_duration_ms = (time.perf_counter() - epoch_start_time) * 1000
    profiler.log_epoch(avg_loss, epoch_duration_ms)

    return state, avg_loss


def eval_model(state, images, labels, batch_size):
    losses = []
    num_batches = len(images) // batch_size

    for i in range(num_batches):
        batch_images, batch_labels = get_batch(images, labels, i, batch_size)
        loss, _ = eval_step(state, batch_images, batch_labels)
        losses.append(loss.item())

    return np.mean(losses)


def main(
    train: bool,
    compile: bool,
    batch_size: int = 64,
    num_epochs: int = 2,
    learning_rate: float = 0.1,
    device: str = "cpu",
    model_variant: str = "simple",
    scale_data: bool = True,
):
    if device == "cpu":
        jax.config.update("jax_platform_name", "cpu")

    print(f"Platform: {platform.system()} {platform.release()} (Python {platform.python_version()})")
    print(f"JAX devices: {jax.devices()}")

    data_dir = Path(os.getenv("ZG_DATA_DIR", "data"))
    csv_path = data_dir / "mnist_train_full.csv"

    print(f"{csv_path=}")
    print(f"{train=}")
    print(f"{compile=}")
    print(f"{batch_size=}")
    print(f"{num_epochs=}")
    print(f"{learning_rate=}")
    print(f"{device=}")
    print(f"{model_variant=}")

    # For performance and fairness we load everything up front
    images, labels = load_mnist_all_at_once(csv_path, scale_data)
    num_batches = len(images) // batch_size
    print(f"n_batches={num_batches}")

    # Initialize model and state
    rng = jax.random.PRNGKey(0)
    model = Model(variant=model_variant)
    state = create_train_state(rng, learning_rate, model, input_shape=(batch_size, 28, 28, 1))

    profiler = Profiler(num_batches)
    start_time = time.perf_counter()

    # Compile by running a step
    if compile:
        print("Compiling training step...")
        warmup_images, warmup_labels = get_batch(images, labels, 0, batch_size)
        state, _, _ = train_step(state, warmup_images, warmup_labels)
        print("Compilation complete")

    if not train:
        # Evaluation only
        eval_loss = eval_model(state, images, labels, batch_size)
        print(f"Evaluation loss: {eval_loss:.5f}")
        return

    # Training loop
    for epoch in range(num_epochs):
        profiler.epoch = epoch
        state, avg_loss = train_epoch(state, images, labels, batch_size, profiler)

    total_time_ms = (time.perf_counter() - start_time) * 1000
    print(f"Training complete ({num_epochs} epochs). [{total_time_ms:.2f}ms]")
    profiler.log_final_stats()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", action="store_true", default=False, help="Whether to compile.")
    parser.add_argument("-t", action="store_true", default=False, help="Whether to train the model.")
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "tpu"],
        default="cpu",
        help="Device to use.",
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training.")
    parser.add_argument("--num_epochs", type=int, default=2, help="Number of epochs.")
    parser.add_argument("--learning_rate", type=float, default=0.1, help="Learning rate.")
    parser.add_argument(
        "--model_variant", type=str, choices=["simple", "simple2"], default="simple", help="Model variant to use."
    )

    args = parser.parse_args()

    main(
        train=args.t,
        compile=args.c,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        device=args.device,
        model_variant=args.model_variant,
    )
