import os
import platform
import time
from enum import StrEnum, auto
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class Model(nn.Module):
    def __init__(self, variant="simple"):
        super().__init__()
        if variant == "simple":
            dim_fc1 = (784, 128)
            dim_fc2 = (128, 64)
            dim_fc3 = (64, 10)
        elif variant == "simple2":
            dim_fc1 = (784, 784)
            dim_fc2 = (784, 128)
            dim_fc3 = (128, 10)
        else:
            raise ValueError("Invalid variant")

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(*dim_fc1)
        torch.nn.init.kaiming_normal_(self.fc1.weight)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(*dim_fc2)
        torch.nn.init.kaiming_normal_(self.fc2.weight)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(*dim_fc3)
        torch.nn.init.kaiming_normal_(self.fc3.weight)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.fc3(x)


class ModelAg(nn.Module):
    def __init__(self, variant="simple"):
        super().__init__()
        if variant == "simple":
            dim_fc1 = (784, 128)
            dim_fc2 = (128, 64)
            dim_fc3 = (64, 10)
        elif variant == "simple2":
            dim_fc1 = (784, 784)
            dim_fc2 = (784, 128)
            dim_fc3 = (128, 10)
        else:
            raise ValueError("Invalid variant")

        # Initialize weights and biases
        self.w1 = torch.empty(dim_fc1, requires_grad=True)
        self.b1 = torch.zeros(dim_fc1[1], requires_grad=True)
        self.w2 = torch.randn(dim_fc2, requires_grad=True)
        self.b2 = torch.empty(dim_fc2[1], requires_grad=True)
        self.w3 = torch.randn(dim_fc3, requires_grad=True)
        self.b3 = torch.empty(dim_fc3[1], requires_grad=True)

        torch.nn.init.kaiming_normal_(self.w1)
        torch.nn.init.kaiming_normal_(self.w2)
        torch.nn.init.kaiming_normal_(self.w3)

    def relu(self, x):
        return torch.clamp(x, min=0)

    def flatten(self, x):
        return x.view(x.size(0), -1)

    def forward(self, x):
        x = self.flatten(x)

        # l1
        x = x @ self.w1 + self.b1
        x = self.relu(x)

        # l2
        x = x @ self.w2 + self.b2
        x = self.relu(x)

        # l3
        x = x @ self.w3 + self.b3
        return x

    def parameters(self):
        return [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]


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


class Profiler:
    def __init__(self, total_batches):
        self.epoch = 0
        self.batch = 0
        self.total_time = 0
        self.total_batches = total_batches

    def log(self, loss, duration_ms):
        print(f"train_loss: {loss:.5f} [{self.batch}/{self.total_batches}] [ms/sample: {duration_ms:.4f}]")

    def log_epoch(self, avg_loss, duration_ms):
        print(f"Epoch {self.epoch + 1}: Avg Loss = {avg_loss:.4f} ({duration_ms:.2f}ms)")


def main(
    train: bool,
    compile: bool,
    batch_size: int = 64,
    num_epochs: int = 2,
    learning_rate: float = 0.1,
    device: str = "cpu",
    grad_mode: str = "default",
    model_variant: str = "simple",
    autograd: bool = False,
):
    data_dir = Path(os.getenv("DATA_DIR", "/tmp/zigrad_mnist_data"))
    dataloader = load_mnist(data_dir / "mnist_train_full.csv", batch_size)
    print(f"train={train}")
    print(f"compile={compile}")
    print(f"batch_size={batch_size}")
    print(f"num_epochs={num_epochs}")
    print(f"learning_rate={learning_rate}")
    print(f"device={device}")
    print(f"n_batches={len(dataloader)}")
    print(f"grad_mode={grad_mode}")
    print(f"Platform: {platform.system()} {platform.release()} (Python {platform.python_version()})")
    model = ModelAg(model_variant).to(device) if autograd else Model(model_variant).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    if not train:
        model.eval()
    if compile:
        model = torch.compile(model)

    profiler = Profiler(len(dataloader))

    start_time = time.perf_counter()
    losses = []

    for epoch in range(num_epochs):
        profiler.epoch = epoch
        total_loss = 0
        epoch_start_time = time.perf_counter()

        for i, (images, labels) in enumerate(dataloader):
            profiler.batch = i + 1
            images, labels = images.to(device), labels.to(device)

            batch_start_time = time.perf_counter()

            if train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            losses.append(loss.item())

            if train:
                loss.backward()
                optimizer.step()

            batch_end_time = time.perf_counter()
            duration_ms = (batch_end_time - batch_start_time) * 1000 / batch_size  # ms per sample

            total_loss += loss.item()
            profiler.log(loss.item(), duration_ms)

        avg_loss = total_loss / len(dataloader)
        epoch_duration_ms = (time.perf_counter() - epoch_start_time) * 1000
        profiler.log_epoch(avg_loss, epoch_duration_ms)

    total_time_ms = (time.perf_counter() - start_time) * 1000
    print(f"Training complete ({num_epochs} epochs). [{total_time_ms:.2f}ms]")

    print(f"Loss s={np.std(losses):.5} mu={np.mean(losses):.5}")


class GradMode(StrEnum):
    default = auto()
    nograd = auto()
    inference = auto()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", action="store_true", default=False, help="Whether to torch.compile().")
    parser.add_argument("-t", action="store_true", default=False, help="Whether to train the model.")
    parser.add_argument("--grad_mode", type=GradMode, default="default", help="Set the PyTorch grad tracking policy.")
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
        grad_mode=args.grad_mode,
        model_variant=args.model_variant,
        autograd=args.autograd,
    )
