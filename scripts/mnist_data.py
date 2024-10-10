from pathlib import Path
import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms


def pull_data():
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.MNIST(root="/tmp/zigrad_mnist_data", train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root="/tmp/zigrad_mnist_data", train=False, download=True, transform=transform)
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
    train, test = pull_data()
    train_csv = Path("/tmp/zigrad_test_mnist_train_full.csv")
    test_csv = Path("/tmp/zigrad_test_mnist_test_full.csv")
    create_mnist_csv(train, train_csv)
    # Create some smaller subsets for faster testing
    head(4096, train_csv, train_csv.with_stem("zigrad_test_mnist_train_small"))
    create_mnist_csv(test, test_csv)
    head(512, train_csv, train_csv.with_stem("zigrad_test_mnist_test_small"))
