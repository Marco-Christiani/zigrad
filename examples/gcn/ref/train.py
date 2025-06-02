import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid


class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


if __name__ == "__main__":
    from pathlib import Path

    data_path = Path(__file__).parent.parent / "data"
    dataset = Planetoid(root=str(data_path), name="cora")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GCN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    def train(data):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        return loss.item()

    def test(data):
        model.eval()
        logits = model(data)
        accs = []
        for mask in [data.train_mask, data.val_mask, data.test_mask]:
            pred = logits[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
            accs.append(acc)
        return accs

    for epoch in range(0, 50):
        data = dataset[0]
        loss = train(data)
        train_acc, val_acc, test_acc = test(data)

        print(
            f"Epoch: {epoch + 1:02d}, Loss: {loss:.4f}, "
            f"Train_acc: {train_acc:.2f}, Val_acc: {val_acc:.2f}, Test_acc: {test_acc:.2f}"
        )
