"""Pulls mnist dataset and converts to csv artifacts.

Optimized for the impatient.
"""

import asyncio
import gzip
import http.client
import itertools
import logging
import multiprocessing as mp
import os
import ssl
import struct
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
DATA_DIR = Path(os.getenv("ZG_DATA_DIR", "data"))
SCALE_VALS = bool(os.environ.get("SCALE_VALS", True))  # whether to scale pixel values
CHUNK_SIZE = 1024 * 1024  # 1MB chunks for file operations
MAX_WORKERS = min(32, mp.cpu_count() * 2)


@dataclass(frozen=True)
class ChunkData:
    """Represents an MNIST chunk."""

    image_chunk: bytes
    label_chunk: bytes
    chunk_idx: int
    chunk_size: int


async def download_file(url: str) -> bytes:
    """Download a file asynchronously using http.client."""
    parsed_url = urlparse(url)
    context = ssl.create_default_context()

    loop = asyncio.get_event_loop()

    def _download() -> bytes:
        conn = http.client.HTTPSConnection(parsed_url.netloc, context=context)
        try:
            conn.request("GET", parsed_url.path)
            response = conn.getresponse()
            if response.status != 200:
                raise RuntimeError(f"Failed to download {url}: {response.status} {response.reason}")
            return response.read()
        finally:
            conn.close()

    return await loop.run_in_executor(None, _download)


async def download_mnist_files() -> dict[str, bytes]:
    """Download all MNIST files asynchronously."""
    base_url = "https://ossci-datasets.s3.amazonaws.com/mnist"
    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz",
    }

    download_tasks = {
        name: asyncio.create_task(download_file(f"{base_url}/{filename}")) for name, filename in files.items()
    }

    downloaded = {}
    for name, task in download_tasks.items():
        downloaded[name] = await task

    return downloaded


def extract_gz(content: bytes) -> bytes:
    """Extract gzipped content."""
    with BytesIO(content) as compressed, BytesIO() as decompressed:
        with gzip.GzipFile(fileobj=compressed, mode="rb") as gz:
            decompressed.write(gz.read())
        return decompressed.getvalue()


def process_chunk(chunk_data: ChunkData) -> tuple[int, list[str]]:
    """Process a chunk of images and labels into CSV lines.

    Returns tuple of (chunk_idx, processed_lines) to maintain order.
    """
    scaling_factor = 255 if SCALE_VALS else 1
    lines = []

    # pre-allocate one-hot arrays for all digits
    one_hot_arrays = [["1" if j == i else "0" for j in range(10)] for i in range(10)]

    for i in range(chunk_data.chunk_size):
        label = struct.unpack_from("B", chunk_data.label_chunk, i)[0]
        one_hot = one_hot_arrays[label]

        # read pixels in chunks
        pixel_offset = i * 28 * 28
        pixels = [
            str(p / scaling_factor) for p in struct.unpack_from(f"{28 * 28}B", chunk_data.image_chunk, pixel_offset)
        ]

        # csv line
        lines.append(",".join(one_hot + pixels))

    return chunk_data.chunk_idx, lines


def create_mnist_csv_parallel(image_data: bytes, label_data: bytes, output_path: Path, n_images: int) -> None:
    """Convert MNIST binary data to CSV using parallel processing."""
    # skip headers
    image_data = image_data[16:]
    label_data = label_data[8:]

    chunk_size = max(1024, n_images // MAX_WORKERS)  # Don't want chunks too small
    chunks = []

    # chunk with index for ordering
    for chunk_idx, i in enumerate(range(0, n_images, chunk_size)):
        end_idx = min(i + chunk_size, n_images)
        current_chunk_size = end_idx - i

        chunk = ChunkData(
            image_chunk=image_data[i * 28 * 28 : (i + current_chunk_size) * 28 * 28],
            label_chunk=label_data[i : i + current_chunk_size],
            chunk_idx=chunk_idx,
            chunk_size=current_chunk_size,
        )
        chunks.append(chunk)

    # chunks and maintain order
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_chunk = {executor.submit(process_chunk, chunk): chunk.chunk_idx for chunk in chunks}

        # process results in order
        results: dict[int, list[str]] = {}
        total_chunks = len(chunks)

        for future in progress(as_completed(future_to_chunk), len(future_to_chunk.keys())):
            chunk_idx, chunk_lines = future.result()
            results[chunk_idx] = chunk_lines

            # logging.debug(f"{len(results)}/{total_chunks} chunks")

        with output_path.open("w", buffering=CHUNK_SIZE) as f:
            for chunk_idx in range(total_chunks):
                chunk_lines = results[chunk_idx]
                f.write("\n".join(chunk_lines) + "\n")


def head_efficient(n: int, inpath: Path, outpath: Path) -> None:
    """Create a subset of first n lines using efficient buffering."""
    with inpath.open("r", buffering=CHUNK_SIZE) as src, outpath.open("w", buffering=CHUNK_SIZE) as dst:
        # chunks for better i/o performance
        remaining = n
        while remaining > 0:
            chunk = list(itertools.islice(src, min(1000, remaining)))
            if not chunk:
                break
            dst.write("".join(chunk))
            remaining -= len(chunk)


def progress(obj, total=None, msg=""):
    total = total or len(obj)

    for i in range(1, total + 1):
        sys.stdout.write(f"\r{msg}{i / total:.0%}")
        sys.stdout.flush()
        try:
            yield next(obj)
        except:
            yield obj[i]
    sys.stdout.write("\r")


async def main() -> None:
    """Pull mnist dataset and convert to csv artifacts with optimizations."""
    DATA_DIR.mkdir(exist_ok=True)

    train_csv_full = DATA_DIR / "mnist_train_full.csv"
    test_csv_full = DATA_DIR / "mnist_test_full.csv"
    train_csv_small = DATA_DIR / "mnist_train_small.csv"
    test_csv_small = DATA_DIR / "mnist_test_small.csv"

    if not all(p.exists() for p in (train_csv_full, test_csv_full, train_csv_small, test_csv_small)):
        logging.info("Downloading dataset files")
        downloaded_files = await download_mnist_files()

        logging.info("Extracting gzipped files")
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            extracted_files = {name: executor.submit(extract_gz, content) for name, content in downloaded_files.items()}
            extracted_files = {name: future.result() for name, future in extracted_files.items()}

        logging.info("Converting training set to CSV format")
        create_mnist_csv_parallel(
            extracted_files["train_images"],
            extracted_files["train_labels"],
            train_csv_full,
            60000,
        )

        logging.info("Converting test set to CSV format")
        create_mnist_csv_parallel(
            extracted_files["test_images"],
            extracted_files["test_labels"],
            test_csv_full,
            10000,
        )

        logging.info("Creating smaller subsets")
        head_efficient(4096, train_csv_full, train_csv_small)
        head_efficient(512, test_csv_full, test_csv_small)

        logging.info("Done.")


if __name__ == "__main__":
    logging.basicConfig(
        level=LOG_LEVEL,
        format="[%(levelname)s] %(funcName)s: %(message)s",
    )
    asyncio.run(main())
