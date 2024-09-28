import time
from contextlib import contextmanager
from enum import StrEnum, auto

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.optim as optim
from plotly.subplots import make_subplots
from torch.utils.data import DataLoader, TensorDataset


class Profiler:
    def __init__(self):
        self.data = []
        self.has_cuda = torch.cuda.is_available()
        self.has_mps = torch.backends.mps.is_available()
        self.epoch = 0
        self.batch = 0

    @contextmanager
    def measure(self, name):
        if self.has_cuda:
            torch.cuda.synchronize()
        if self.has_mps:
            torch.mps.synchronize()
        start = time.perf_counter()
        yield
        if self.has_cuda:
            torch.cuda.synchronize()
        if self.has_mps:
            torch.mps.synchronize()
        end = time.perf_counter()
        self.data.append(
            {"type": "duration", "name": name, "value": end - start, "epoch": self.epoch, "batch": self.batch}
        )

    def log_metric(self, name, value):
        self.data.append({"type": "metric", "name": name, "value": value, "epoch": self.epoch, "batch": self.batch})

    def set_epoch(self, epoch):
        self.epoch = epoch

    def set_batch(self, batch):
        self.batch = batch

    def save(self, filename):
        df = pd.DataFrame(self.data)
        df.to_csv(filename, index=False)

    @staticmethod
    def plot(filename, batch_size):
        df = pd.read_csv(filename)
        dur_df = df.query("type == 'duration' and name == 'step'")
        loss_df = df.query("type == 'metric' and name == 'loss'")

        fig = make_subplots(rows=2, cols=1, subplot_titles=("Loss per Batch", "MS per Sample"))

        # Plot Loss
        fig.add_trace(
            go.Scatter(x=loss_df["epoch"] * loss_df["batch"], y=loss_df["value"], mode="lines", name="Loss"),
            row=1,
            col=1,
        )

        # Plot MS per Sample
        fig.add_trace(
            go.Scatter(
                x=dur_df["epoch"] * dur_df["batch"],
                y=dur_df["value"] * 1000 / batch_size,  # Convert to ms per sample
                mode="lines",
                name="MS per Sample",
            ),
            row=2,
            col=1,
        )

        fig.update_layout(height=800, width=1000, title_text="PyTorch Performance")
        fig.write_html("pytorch_profile.html")


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)

        torch.nn.init.kaiming_normal_(self.fc1.weight)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        torch.nn.init.kaiming_normal_(self.fc2.weight)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 10)
        torch.nn.init.kaiming_normal_(self.fc3.weight)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.fc3(x)


def load_mnist(filepath, batch_size):
    print(f"Loading data from {filepath}")
    data = np.loadtxt(filepath, delimiter=",")
    print(f"Data shape: {data.shape}")

    if data.shape[1] != 794:  # 784 pixels + 10 one-hot label
        raise ValueError(f"Unexpected data shape. Expected 794 columns, got {data.shape[1]}")

    images = torch.FloatTensor(data[:, 10:])
    labels = torch.FloatTensor(data[:, :10])  # one-hot labels

    print(f"Images shape: {images.shape}, Labels shape: {labels.shape}")

    images = images.view(-1, 1, 28, 28)  # (batch, channels, height, width)
    print(f"Reshaped images shape: {images.shape}")

    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def main(
    train: bool,
    compile: bool,
    batch_size: int = 64,
    num_epochs: int = 2,
    learning_rate: float = 0.1,
    device: str = "cpu",
    grad_mode: str = "default",
):
    dataloader = load_mnist("/tmp/zigrad_test_mnist_train_full.csv", batch_size)
    print(f"{train=}")
    print(f"{compile=}")
    print(f"{batch_size=}")
    print(f"{num_epochs=}")
    print(f"{learning_rate=}")
    print(f"{device=}")
    print(f"n_batches={len(dataloader)}")
    print(f"grad_mode={grad_mode}")
    mode = "train" if train else "infer"

    model = Model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Handle gradient modes
    grad_context = None
    if grad_mode == "nograd":
        grad_context = torch.no_grad()
    elif grad_mode == "inference":
        grad_context = torch.inference_mode()
    else:
        grad_context = torch.enable_grad()  # This is the default behavior

    if not train:
        model.eval()
    if compile:
        model = torch.compile(model)

    ns_per_ms = 1e6
    durs = []

    profiler = Profiler()

    s = time.monotonic_ns()

    with grad_context:
        for epoch in range(num_epochs):
            profiler.set_epoch(epoch)
            with profiler.measure("epoch"):
                total_loss = 0
                et0 = time.monotonic_ns()
                for i, (images, labels) in enumerate(dataloader):
                    profiler.set_batch(i)
                    with profiler.measure("step"):
                        t0 = time.monotonic_ns()
                        images, labels = images.to(device), labels.to(device)
                        if train:
                            optimizer.zero_grad()
                        with profiler.measure("fwd"):
                            outputs = model(images)
                        # NOTE: including this in fwd depends on how its timed in zigrad
                        loss = criterion(outputs, labels)
                        if train:
                            with profiler.measure("bwd"):
                                loss.backward()
                            # NOTE: including this in fwd depends on how its timed in zigrad
                            optimizer.step()
                        total_loss += loss.item()
                        dur_ns = time.monotonic_ns() - t0
                        durs.append(dur_ns)
                        profiler.log_metric("loss", loss.item())
                        dur_ms = dur_ns / ns_per_ms
                        print(
                            f" [{i}/{len(dataloader)}] {mode}_loss: {loss.item():.5f} [{dur_ms:.4f}, {dur_ms/batch_size:.4f}/sample]"
                        )

                avg_loss = total_loss / len(dataloader)
                edur_ms = (time.monotonic_ns() - et0) / ns_per_ms
                print(f"Epoch {epoch + 1}: Avg Loss = {avg_loss:.4f} ({edur_ms})")

    # durs = durs[1:]  # ignore compilation
    e = time.monotonic_ns()
    total = (e - s) / ns_per_ms
    durs = np.array(durs) / ns_per_ms
    durs_batch = durs / batch_size
    print(f"Done. [{train=} {compile=}]")
    print(
        f"Per sample (mu={durs.mean():.5f} std={durs.std():.5f} min={durs.min():.5f} max={durs.max():.5f} total={total:.5f})"
    )
    profiler.save("/tmp/zg_torch_profile_data.csv")
    Profiler.plot("/tmp/zg_torch_profile_data.csv", batch_size)


class GradMode(StrEnum):
    default = auto()
    nograd = auto()
    inference = auto()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", type=bool, default=False, help="Whether to torch.compile().")
    parser.add_argument("-t", type=bool, default=False, help="Whether to train the model.")
    parser.add_argument("--grad_mode", type=GradMode, default="default", help="Set the PyTorch grad tracking policy.")
    parser.add_argument(
        "--device", type=str, choices=["cpu", "cuda", "mps"], default="cpu", help="Device to use for training."
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training.")
    parser.add_argument("--num_epochs", type=int, default=2, help="Number of epochs.")
    parser.add_argument("--learning_rate", type=float, default=0.1, help="Learning rate.")

    args = parser.parse_args()

    main(
        train=args.t,
        compile=args.c,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        device=args.device,
        grad_mode=args.grad_mode,
    )
