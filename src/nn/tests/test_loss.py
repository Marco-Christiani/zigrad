import pathlib
from typing import cast
import torch
import torch.nn.functional as F
import json


def _smce(input_tensor, target_tensor, dim) -> dict[str, list | float]:
    softmax_output = F.softmax(input_tensor, dim=dim)
    loss = F.cross_entropy(
        input_tensor.view(-1, input_tensor.shape[-1]),
        target_tensor.view(-1, target_tensor.shape[-1]),
    )
    loss.backward()

    return {
        "shape": list(input_tensor.shape),
        "dim": dim,
        "input": input_tensor.flatten().tolist(),
        "target": target_tensor.flatten().tolist(),
        "softmax_output": softmax_output.detach().flatten().tolist(),
        "loss": cast(float, loss.item()),
        "input_grad": input_tensor.grad.flatten().tolist(),
    }


def generate_smce_tests(output_dir: pathlib.Path) -> None:
    test_cases = {
        "softmax_crossentropy_1d": _smce(
            torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True), torch.tensor([0.0, 1.0, 0.0, 0.0]), 0
        ),
        "softmax_crossentropy_2d": _smce(
            torch.tensor([[1.0, 2.0, 3.0], [4.0, 1.0, 2.0]], requires_grad=True),
            torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            1,
        ),
        "softmax_crossentropy_3d": _smce(
            torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], requires_grad=True),
            torch.tensor([[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]]),
            2,
        ),
    }

    dest = output_dir.joinpath("softmax_crossentropy_test_cases.json")
    dest.write_text(json.dumps(obj=test_cases, indent=2))


if __name__ == "__main__":
    generate_smce_tests(pathlib.Path("/tmp"))
