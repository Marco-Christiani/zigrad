import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


class SimpleMNISTModel(nn.Module):
    def __init__(self):
        super(SimpleMNISTModel, self).__init__()
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
        x = self.fc3(x)
        return x


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


def main(inference: bool, compile: bool):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = SimpleMNISTModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    batch_size = 64
    num_epochs = 2
    dataloader = load_mnist("/tmp/zigrad_test_mnist_train_full.csv", batch_size)
    print(f"Number of batches: {len(dataloader)}")
    if inference:
        model = model.eval()
    if compile:
        model.compile()
    ns_per_ms = 1000**2
    durs = []
    s = time.monotonic_ns()
    for epoch in range(num_epochs):
        total_loss = 0
        et0 = time.monotonic_ns()
        for i, (images, labels) in enumerate(dataloader):
            t0 = time.monotonic_ns()
            images, labels = images.to(device), labels.to(device)
            if not inference:
                optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            if not inference:
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
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
        f"Done. [{inference=} {compile=}] (mu={durs.mean():.5f} std={durs.std():.5f} min={durs.min():.5f} max={durs.max():.5f} total={durs.sum():.5f})"
    )


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", type=bool, default=False)
    parser.add_argument("-e", type=bool, default=False)
    args = parser.parse_args()

    main(args.e, args.c)
