import os
from pathlib import Path
import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms


def pull_data(output_dir: Path):
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.MNIST(
        root=output_dir, train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.MNIST(
        root=output_dir, train=False, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)
    return trainloader, testloader


def create_mnist_csv(dataloader: torch.utils.data.DataLoader, output_path: Path):
    with output_path.open("w") as f:
        for data in dataloader:
            images, labels = data

            label_one_hot = [0] * 10
            label_one_hot[labels.item()] = 1
            label_str = ",".join(map(str, label_one_hot))

            image_flat = images.numpy().flatten()
            image_str = ",".join(map(str, image_flat))

            f.write(f"{label_str},{image_str}\n")


def head(n: int, inpath: Path, outpath: Path):
    with inpath.open("r") as src, outpath.open("w") as dst:
        i = 0
        while i < n and (line := src.readline()):
            dst.write(line)
            i += 1


if __name__ == "__main__":
    output_dir = Path(os.getenv("ZG_TEST_DATA_DIR", "/tmp/zigrad_mnist_data"))
    output_dir.mkdir(parents=False, exist_ok=True)
    train, test = pull_data(output_dir)

    # Save full and some smaller subsets for faster testing
    train_csv = output_dir / "zigrad_test_mnist_train_full.csv"
    test_csv = output_dir / "zigrad_test_mnist_test_full.csv"
    print("creating", str(train_csv), str(train_csv.absolute()))
    create_mnist_csv(train, train_csv)
    head(4096, train_csv, train_csv.with_stem("zigrad_test_mnist_train_small"))
    print("creating", str(test_csv))
    create_mnist_csv(test, test_csv)
    head(512, test_csv, test_csv.with_stem("zigrad_test_mnist_test_small"))
