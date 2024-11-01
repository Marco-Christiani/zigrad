"""Pulls mnist dataset and converts to csv artifacts."""

import gzip
import logging
import os
import struct
import urllib.request
from pathlib import Path
from typing import Tuple

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="[%(levelname)s] %(funcName)s: %(message)s")
DATA_DIR = Path(os.getenv("ZG_DATA_DIR", "data"))
SCALE_VALS = bool(os.environ.get("SCALE_VALS", True))  # torch will underperform without scaling


def _download_extract(url: str, output_path: Path) -> None:
    """Download and extract a gzipped file."""
    gz_path = output_path.with_suffix(".gz")
    urllib.request.urlretrieve(url, gz_path)
    with gzip.open(gz_path, "rb") as gz_file, output_path.open("wb") as out_file:
        out_file.write(gz_file.read())
    gz_path.unlink()  # dont need gz anymore


def _pull_mnist(output_dir: Path) -> Tuple[Path, Path, Path, Path]:
    """Download and extract MNIST data."""
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    base_url = "https://ossci-datasets.s3.amazonaws.com/mnist"
    files = {
        "train_images": ("train-images-idx3-ubyte", 60000),
        "train_labels": ("train-labels-idx1-ubyte", 60000),
        "test_images": ("t10k-images-idx3-ubyte", 10000),
        "test_labels": ("t10k-labels-idx1-ubyte", 10000),
    }

    paths = {}
    for name, (filename, _) in files.items():
        output_path = raw_dir / filename
        if not output_path.exists():
            url = f"{base_url}/{filename}.gz"
            logging.info(f"Dowloading {url} to {output_path!s}")
            _download_extract(url, output_path)
        paths[name] = output_path

    return (paths["train_images"], paths["train_labels"], paths["test_images"], paths["test_labels"])


def create_mnist_csv(image_path: Path, label_path: Path, output_path: Path, n_images: int) -> None:
    """Convert MNIST binary files to CSV with one-hot encoded labels."""
    scaling_factor = 255 if SCALE_VALS else 1
    if not SCALE_VALS:
        logging.warning("Not scaling values, pytorch will underperform.")
    logging.info(f"Creating {output_path!s}")
    with image_path.open("rb") as f_img, label_path.open("rb") as f_lbl, output_path.open("w") as f_out:
        # skip headers
        f_img.read(16)
        f_lbl.read(8)

        for _ in range(n_images):
            # read and encode label
            label = struct.unpack("B", f_lbl.read(1))[0]
            one_hot = ["1" if i == label else "0" for i in range(10)]

            # read image pixels
            pixels = [str(struct.unpack("B", f_img.read(1))[0] / scaling_factor) for _ in range(28 * 28)]

            # csv line
            f_out.write(",".join(one_hot + pixels) + "\n")


def head(n: int, inpath: Path, outpath: Path) -> None:
    """Create a subset of first n lines."""
    logging.info(f"Creating {outpath!s}")
    with inpath.open("r") as src, outpath.open("w") as dst:
        for _ in range(n):
            if line := src.readline():
                dst.write(line)


def main() -> None:
    """Pull mnist dataset and converts to csv artifacts."""
    DATA_DIR.mkdir(exist_ok=True)

    train_csv_full = DATA_DIR / "mnist_train_full.csv"
    test_csv_full = DATA_DIR / "mnist_test_full.csv"
    train_csv_small = DATA_DIR / "mnist_train_small.csv"
    test_csv_small = DATA_DIR / "mnist_test_small.csv"

    if not all(p.exists() for p in (train_csv_full, test_csv_full, train_csv_small, test_csv_small)):
        # download and convert csv
        logging.info("Downloading dataset")
        train_images, train_labels, test_images, test_labels = _pull_mnist(DATA_DIR)
        logging.info("Converting to csv")
        create_mnist_csv(train_images, train_labels, train_csv_full, 60000)
        create_mnist_csv(test_images, test_labels, test_csv_full, 10000)

        # create smaller subsets (faster tests)
        head(4096, train_csv_full, train_csv_small)
        head(512, test_csv_full, test_csv_small)


if __name__ == "__main__":
    main()
