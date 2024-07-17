import pathlib
from typing import cast
import torch
import torch.nn as nn
import json


def _smce(input_tensor, target_tensor) -> dict[str, list | float]:
    cel = nn.CrossEntropyLoss()
    loss = cel(input_tensor, target_tensor)
    # loss = F.binary_cross_entropy_with_logits(
    #     softmax_output,
    #     # input_tensor,  # .view(-1, input_tensor.shape[-1]),
    #     target_tensor,  # .view(-1, target_tensor.shape[-1]),
    # )
    loss.backward(retain_graph=True)

    return {
        "shape": list(input_tensor.shape),
        "input": input_tensor.flatten().tolist(),
        "target": target_tensor.flatten().tolist(),
        "loss": cast(float, loss.item()),
        "input_grad": input_tensor.grad.flatten().tolist(),
    }


def generate_smce_tests(output_dir: pathlib.Path) -> None:
    test_cases = {
        "softmax_crossentropy_1d": _smce(
            torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True),
            torch.tensor([0.0, 1.0, 0.0, 0.0]),
        ),
        "softmax_crossentropy_2d": _smce(
            torch.tensor([[1.0, 2.0, 3.0], [4.0, 1.0, 2.0]], requires_grad=True),
            torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        ),
    }

    dest = output_dir.joinpath("softmax_crossentropy_test_cases.json")
    dest.write_text(json.dumps(obj=test_cases, indent=2))


if __name__ == "__main__":
    generate_smce_tests(pathlib.Path("/tmp"))
