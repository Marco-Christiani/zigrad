from enum import StrEnum, auto
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


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

    ns_per_ms = 1000**2
    durs = []
    s = time.monotonic_ns()

    with grad_context:  # Apply gradient mode context
        for epoch in range(num_epochs):
            total_loss = 0
            et0 = time.monotonic_ns()
            for i, (images, labels) in enumerate(dataloader):
                t0 = time.monotonic_ns()
                images, labels = images.to(device), labels.to(device)
                if train:
                    optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                if train:
                    loss.backward()
                    optimizer.step()
                total_loss += loss.item()
                dur_ns = time.monotonic_ns() - t0
                dur_ms = dur_ns / ns_per_ms
                durs.append(dur_ns / batch_size)
                print(f"Loss: {loss.item():.5f} {i/len(dataloader):.2f} [{i}/{len(dataloader)}] [{dur_ms/batch_size}]")

            avg_loss = total_loss / len(dataloader)
            edur_ms = (time.monotonic_ns() - et0) / ns_per_ms
            print(f"Epoch {epoch + 1}: Avg Loss = {avg_loss:.4f} ({edur_ms})")

    durs = durs[1:]  # ignore compilation
    e = time.monotonic_ns()
    print((e - s) / (1000**3))
    durs = np.array(durs) / ns_per_ms
    print(
        f"Done. [{train=} {compile=}] (mu={durs.mean():.5f} std={durs.std():.5f} min={durs.min():.5f} max={durs.max():.5f} total={durs.sum():.5f})"
    )


class GradMode(StrEnum):
    default = auto()
    nograd = auto()
    inference = auto()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", type=bool, default=False, help="Whether to torch.compile().")
    parser.add_argument("-t", type=bool, default=True, help="Whether to train the model.")
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
