# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch",
#     "torch_geometric",
#     "numpy",
#     "scipy",
# ]
# ///

from pathlib import Path
import os

import torch
import numpy as np
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data


def dump_to_csv(data: Data, out_path_papers: Path, out_path_cities: Path) -> None:
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
        out_path_papers,
        papers.numpy(),
        header="y,train_mask,eval_mask,test_mask,x...",
        delimiter=",",
        fmt="%d",
    )

    np.savetxt(
        out_path_cities,
        data.edge_index.transpose(0, 1).numpy(),
        delimiter=",",
        fmt="%d",
    )


if __name__ == "__main__":
    ZG_DATA_DIR = os.environ.get("ZG_DATA_DIR")
    if not ZG_DATA_DIR or not Path(ZG_DATA_DIR).exists():
        ZG_DATA_DIR = Path(os.environ.get("ZG_DATA_DIR", Path(__file__).parent.parent.joinpath("data")))
    else:
        ZG_DATA_DIR = Path(ZG_DATA_DIR)
    out_dir_cora = ZG_DATA_DIR / "cora"
    out_dir_csv = out_dir_cora / "csv"
    out_paths = (out_dir_csv / "papers.csv", out_dir_csv / "cities.csv")
    if all(map(Path.exists, out_paths)):
        exit()

    print(ZG_DATA_DIR)
    print(out_dir_cora)
    print(out_dir_csv)
    out_dir_cora = ZG_DATA_DIR / "cora"
    out_dir_csv.mkdir(exist_ok=True, parents=True)
    dataset = Planetoid(root=str(ZG_DATA_DIR), name="cora")

    # dataset.download()
    dump_to_csv(dataset[0], *out_paths)
