# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch",
#     "torch-geometric",
# ]
# ///

import time
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data


class GCN(torch.nn.Module):
    def __init__(self, num_features: int, num_classes: int):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, data: torch.Tensor):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    args = parser.parse_args()

    data_path = Path(__file__).parent.parent / "data"
    dataset = Planetoid(root=str(data_path), name="cora")
    device = torch.device(args.device)

    model = GCN(dataset.num_features, dataset.num_classes).to(device)
    data = dataset[0].to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    def train():
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        return loss.item()

    def test():
        model.eval()
        logits = model(data)
        accs = []
        for mask in [data.train_mask, data.val_mask, data.test_mask]:
            pred = logits[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
            accs.append(acc)
        return accs

    class PerfTimer:
        def __init__(self, name=None):
            self.name = name

        def __enter__(self):
            self.start = time.perf_counter()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.end = time.perf_counter()
            self.duration = (self.end - self.start) * 1000

    total_train_time = 0
    total_test_time = 0

    for epoch in range(0, 50):
        with PerfTimer("train") as train_timer:
            loss = train()
        with PerfTimer("test") as test_timer:
            train_acc, val_acc, test_acc = test()

        total_train_time += train_timer.duration
        total_test_time += test_timer.duration
        print(
            f"Epoch: {epoch + 1:02d}, Loss: {loss:.4f}, "
            f"Train_acc: {train_acc:.2f}, Val_acc: {val_acc:.2f}, Test_acc: {test_acc:.2f}, "
            f"Train_time {train_timer.duration:.2f} ms, Test_time {test_timer.duration:.2f} ms"
        )

    print(
        f"Avg epoch train time: {total_train_time / 50:.2f} ms, Avg epoch test time: {total_test_time / 50:.2f} ms"
    )
    print(
        f"Total train time: {total_train_time:.2f} ms, Total test time: {total_test_time:.2f} ms"
    )
