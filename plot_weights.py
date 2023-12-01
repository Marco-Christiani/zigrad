from __future__ import annotations

import json
import uuid
import os
import pathlib
from PIL import Image
import numpy as np
import matplotlib.colors
from matplotlib import colormaps
from graphviz import Digraph
import imageio.v2 as imageio


def trace(
    root_node: dict[str, any]
) -> tuple[set[str], list[any], set[tuple[str, str]]]:
    node_labels, node_values, edges = set(), list(), set()

    def build(node: dict[str, any]) -> None:
        if not node["label"]:
            node["label"] = str(uuid.uuid4()).split("-")[0]
        if node["label"] not in node_labels:
            node_labels.add(node["label"])
            node_values.append(node["value"])
            for child in node.get("children", []):
                edges.add((node["label"], child["label"]))
                build(child)

    build(root_node)
    return node_labels, node_values, edges


def _find_node(label: str, root_node: dict[str, any]) -> dict[str, any] | None:
    if root_node["label"] == label:
        return root_node
    for child in root_node.get("children", []):
        result = _find_node(label, child)
        if result:
            return result
    return None


def draw_dot(
    json_data: dict[str, any],
    epoch: int,
    output_directory: str | pathlib.Path,
    global_min: int | float,
    global_max: int | float,
    format: str = "png",
    rankdir: str = "RL",
) -> None:
    nodes, _, edges = trace(json_data)
    # RL and dir=back reverses the graph since we have a topo sort/backprop graph
    dot = Digraph(
        format=format, graph_attr={"rankdir": rankdir}, edge_attr={"dir": "back"}
    )

    for n in nodes:
        node_data = _find_node(n, json_data)
        if node_data is None:
            continue  # Skip if node data is not found

        color = value_to_color(node_data["value"], global_min, global_max)
        color_hex = matplotlib.colors.to_hex(color)

        label = "{ %s | value: %.4f | grad: %.4f }" % (
            n,
            node_data["value"],
            node_data["grad"],
        )
        dot.node(
            name=n, label=label, shape="record", style="filled", fillcolor=color_hex
        )

    for n1, n2 in edges:
        dot.edge(n1, n2)

    file_path = os.path.join(output_directory, f"epoch_{epoch}")
    dot.render(file_path, view=False)


def draw_dot_per_epoch(
    json_data_list: list[dict[str, any]],
    output_directory: str,
    format: str = "png",
    rankdir: str = "RL",
) -> None:
    # Ensure the output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Find global min and max values for normalization
    global_min_value, global_max_value = global_min_max(json_data_list)

    # Generate the images for each epoch
    for epoch, json_data in enumerate(json_data_list, start=1):
        draw_dot(
            json_data,
            epoch,
            output_directory,
            global_min_value,
            global_max_value,
            format,
            rankdir,
        )


def value_to_color(
    value: float | int, min_value: float | int, max_value: float | int
) -> matplotlib.colors.Colormap:
    n = value - min_value
    d = max_value - min_value
    normalized = n / d if d else d
    return colormaps.get_cmap("coolwarm")(normalized)


def global_min_max(json_data_list: list[dict]) -> tuple[int | float, int | float]:
    all_values = []
    for epoch_data in json_data_list:
        labels, values, _ = trace(epoch_data)
        for l, v in zip(labels, values):
            print(f"{l=} {v=}")
            all_values.append(v)
    return min(all_values), max(all_values)


def resize_and_pad(
    image_path: str | pathlib.Path,
    target_size: tuple[int, int] = (1000, 400),
    background_color: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    with Image.open(image_path) as img:
        # Resize the image, preserving the aspect ratio
        # Using LANCZOS resampling
        img.thumbnail(target_size, Image.Resampling.LANCZOS)

        # Create a new image with the target size and background color
        padded_img = Image.new("RGB", target_size, background_color)

        # Calculate the position to paste the resized image onto the background
        paste_position = (
            (target_size[0] - img.size[0]) // 2,
            (target_size[1] - img.size[1]) // 2,
        )
        padded_img.paste(img, paste_position)

        return np.array(padded_img)


def main(json_file: str, output_dir: str = ".") -> None:
    fpath = pathlib.Path(json_file).resolve()
    assert fpath.exists()
    output_dir = pathlib.Path(output_dir).resolve()
    assert output_dir.exists()

    with open(fpath, "r") as f:
        data = json.load(f)
    if isinstance(data, dict):
        print("Plotting for a single graph.")
        min_val, max_val = global_min_max([data])
        draw_dot(data, 0, output_dir, min_val, max_val)
        print("Done")
    elif isinstance(data, list):
        print("Plotting across each epoch")
        draw_dot_per_epoch(data, output_dir)
        print("Building animation")
        filenames = sorted(
            [
                os.path.join(output_dir, f)
                for f in os.listdir(output_dir)
                if f.endswith(".png")
            ]
        )

        standardized_images = [resize_and_pad(image_file) for image_file in filenames]
        imageio.mimsave("model-evolution.gif", standardized_images, fps=1)
        print("Done")


def testit() -> None:
    def data(n: int) -> list[dict[str, any]]:
        return [
            {
                "label": "err",
                "value": np.random.randn(),
                "grad": 1.0e00,
                "children": [
                    {"label": "y", "value": np.random.randn(), "grad": 1.0e00},
                    {
                        "label": "p",
                        "value": np.random.randn(),
                        "grad": -1.0e00,
                        "children": [
                            {
                                "label": "t",
                                "value": np.random.randn(),
                                "grad": -1.0e00,
                                "children": [
                                    {
                                        "label": "x",
                                        "value": np.random.randn(),
                                        "grad": -1.0e00,
                                    },
                                    {
                                        "label": "w",
                                        "value": np.random.randn(),
                                        "grad": -2.0e00,
                                    },
                                ],
                            },
                            {"label": "b", "value": np.random.randn(), "grad": -1.0e00},
                        ],
                    },
                ],
            }
            for _ in range(n)
        ]

    root = data(10)

    output_directory = "outputs/"
    print("generating images")
    draw_dot_per_epoch(root, output_directory)

    print("building animation")
    filenames = sorted(
        [
            os.path.join(output_directory, f)
            for f in os.listdir(output_directory)
            if f.endswith(".png")
        ]
    )

    standardized_images = [resize_and_pad(image_file) for image_file in filenames]
    imageio.mimsave("outputs/model-evolution.gif", standardized_images, fps=1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    parser.add_argument("output-dir")
    args = parser.parse_args()
    main(args.file, args.output_dir)
