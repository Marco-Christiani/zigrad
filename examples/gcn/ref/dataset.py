# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch",
#     "torch_geometric",
#     "numpy",
# ]
# ///

from pathlib import Path

import torch
import numpy as np
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data


def dump_to_csv(data: Data, dir: Path):
    dir.mkdir(exist_ok=True)

    papers = torch.cat(
        (
            data.y.view(-1, 1),
            data.train_mask.view(-1, 1),
            data.val_mask.view(-1, 1),
            data.test_mask.view(-1, 1),
            data.x,
        ),
        dim=1,
    )
    np.savetxt(
        dir / "papers.csv",
        papers.numpy(),
        header="y,train_mask,eval_mask,test_mask,x...",
        delimiter=",",
        fmt="%d",
    )

    np.savetxt(
        dir / "cites.csv",
        data.edge_index.transpose(0, 1).numpy(),
        delimiter=",",
        fmt="%d",
    )


if __name__ == "__main__":
    data_path = Path(__file__).parent.parent / "data"
    dataset = Planetoid(root=str(data_path), name="cora")

    # dataset.download()
    dump_to_csv(dataset[0], data_path / "cora" / "csv")
