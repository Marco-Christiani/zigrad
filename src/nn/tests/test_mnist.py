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
    # for i in range(10):
    #     print(i, labels[i, :10])
    # __import__("pdb").set_trace()
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = SimpleMNISTModel().to(device)
    # criterion = nn.NLLLoss2d()
    # criterion = nn.BCEWithLogitsLoss()
    # criterion = nn.BCELoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    batch_size = 8
    num_epochs = 2
    dataloader = load_mnist("/tmp/mnist_train.csv", batch_size)
    print(f"Number of batches: {len(dataloader)}")

    for epoch in range(num_epochs):
        total_loss = 0
        for i, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            # print(f"Batch {i}: images shape: {images.shape}, labels shape: {labels.shape}")

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            # print("outputs:", outputs[0])
            # print("labels:", labels[0])
            # print("loss:", criterion(outputs[0], labels[0]))
            # return
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            total_loss += loss.item()
            print(f"Loss: {loss.item():.5f} {i/len(dataloader):.2f} [{i}/{len(dataloader)}]")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}: Avg Loss = {avg_loss:.4f}")

    print("Training completed.")


if __name__ == "__main__":
    train()
