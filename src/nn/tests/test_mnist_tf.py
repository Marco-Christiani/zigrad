# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy",
#     "tensorflow",
# ]
# ///
import argparse
import os
import platform
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

physical_devices = tf.config.list_physical_devices("GPU")
if physical_devices:
    tf.config.experimental.set_visible_devices(physical_devices, "GPU")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# policy = mixed_precision.Policy("mixed_float16")
# mixed_precision.set_global_policy(policy)


class Model(keras.Model):
    def __init__(self, variant="simple"):
        super(Model, self).__init__()
        if variant == "simple":
            dim_fc1, dim_fc2, dim_fc3 = 128, 64, 10
        elif variant == "simple2":
            dim_fc1, dim_fc2, dim_fc3 = 784, 128, 10
        else:
            raise ValueError("Invalid variant")

        self.flatten = keras.layers.Flatten()
        self.fc1 = keras.layers.Dense(dim_fc1, activation="relu", kernel_initializer="he_normal")
        self.fc2 = keras.layers.Dense(dim_fc2, activation="relu", kernel_initializer="he_normal")
        self.fc3 = keras.layers.Dense(dim_fc3, dtype="float32")  # Ensure output is float32

    def call(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return self.fc3(x)


def load_mnist(filepath, batch_size):
    print(f"Loading data from {filepath}")
    data = np.loadtxt(filepath, delimiter=",")
    print(f"Data shape: {data.shape}")

    if data.shape[1] != 794:  # 784 pixels + 10 one-hot label
        raise ValueError(f"Unexpected data shape. Expected 794 columns, got {data.shape[1]}")

    images = data[:, 10:].astype(np.float32)
    labels = data[:, :10].astype(np.float32)  # one-hot labels

    images = images.reshape(-1, 28, 28, 1)  # (batch, height, width, channels)
    print(f"Reshaped images shape: {images.shape}")

    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    return dataset.batch(batch_size).shuffle(buffer_size=1024)


class Profiler:
    def __init__(self, total_batches):
        self.epoch = 0
        self.batch = 0
        self.total_batches = total_batches

    def log(self, loss, duration_ms):
        print(f"train_loss: {loss:.5f} [{self.batch}/{self.total_batches}] [ms/sample: {duration_ms:.4f}]")

    def log_epoch(self, avg_loss, duration_ms):
        print(f"Epoch {self.epoch + 1}: Avg Loss = {avg_loss:.4f} ({duration_ms:.2f}ms)")


@tf.function
def train_step(model, images, labels, optimizer, criterion):
    with tf.GradientTape() as tape:
        outputs = model(images)
        loss = criterion(labels, outputs)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


def main(
    train: bool,
    compile: bool,
    batch_size: int = 64,
    num_epochs: int = 2,
    learning_rate: float = 0.1,
    device: str = "cpu",
    # grad_mode: str = "default",
    model_variant: str = "simple",
    autograd: bool = False,
):
    data_dir = Path(os.getenv("ZG_DATA_DIR", "data"))
    csv_path = data_dir / "mnist_train_full.csv"
    dataloader = load_mnist(csv_path, batch_size)
    print(f"train={train}")
    print(f"compile={compile}")
    print(f"batch_size={batch_size}")
    print(f"num_epochs={num_epochs}")
    print(f"learning_rate={learning_rate}")
    print(f"device={device}")
    print(f"n_batches={len(list(dataloader))}")
    # print(f"grad_mode={grad_mode}")
    print(f"Platform: {platform.system()} {platform.release()} (Python {platform.python_version()})")

    model = ModelAg(model_variant) if autograd else Model(model_variant)
    optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
    criterion = keras.losses.CategoricalCrossentropy(from_logits=True)

    profiler = Profiler(len(list(dataloader)))

    start_time = time.perf_counter()
    losses = []

    for epoch in range(num_epochs):
        profiler.epoch = epoch
        total_loss = 0
        epoch_start_time = time.perf_counter()

        for i, (images, labels) in enumerate(dataloader):
            profiler.batch = i + 1

            batch_start_time = time.perf_counter()

            if train:
                with tf.GradientTape() as tape:
                    outputs = model(images)
                    loss = criterion(labels, outputs)
                    losses.append(loss.numpy())

                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            else:
                outputs = model(images)
                loss = criterion(labels, outputs)
                losses.append(loss.numpy())

            batch_end_time = time.perf_counter()
            duration_ms = (batch_end_time - batch_start_time) * 1000 / batch_size  # ms per sample

            total_loss += loss.numpy()
            profiler.log(loss.numpy(), duration_ms)

        avg_loss = total_loss / len(list(dataloader))
        epoch_duration_ms = (time.perf_counter() - epoch_start_time) * 1000
        profiler.log_epoch(avg_loss, epoch_duration_ms)

    total_time_ms = (time.perf_counter() - start_time) * 1000
    print(f"Training complete ({num_epochs} epochs). [{total_time_ms:.2f}ms]")

    print(f"Loss s={np.std(losses):.5} mu={np.mean(losses):.5}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", action="store_true", default=False, help="Whether to torch.compile().")
    parser.add_argument("-t", action="store_true", default=False, help="Whether to train the model.")
    # parser.add_argument("--grad_mode", type=GradMode, default="default", help="Set the PyTorch grad tracking policy.")
    parser.add_argument(
        "--device", type=str, choices=["cpu", "cuda", "mps"], default="cpu", help="Device to use for training."
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training.")
    parser.add_argument("--num_epochs", type=int, default=2, help="Number of epochs.")
    parser.add_argument("--learning_rate", type=float, default=0.1, help="Learning rate.")
    parser.add_argument(
        "--model_variant",
        type=str,
        choices=["simple", "simple2"],
        default="simple",
        help="Which model arch to use.",
    )
    parser.add_argument(
        "--autograd",
        action="store_true",
        default=False,
        help="Use autograd implementation or dedicated modules.",
    )

    args = parser.parse_args()

    main(
        train=args.t,
        compile=args.c,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        device=args.device,
        # grad_mode=args.grad_mode,
        model_variant=args.model_variant,
        autograd=args.autograd,
    )
