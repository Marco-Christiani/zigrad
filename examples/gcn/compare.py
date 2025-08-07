# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "plotly",
# ]
# ///

from pathlib import Path
import json
import plotly.graph_objects as go


def load_avg_trimmed_times(path: Path) -> tuple[float, float]:
    """Load average trimmed train and test times from a JSON file."""
    data = json.loads(path.read_text())
    return (
        data["avg_epoch_train_fbs_trimmed_ms"],
        data["avg_epoch_test_fbs_trimmed_ms"],
    )


zig_json_path = Path(__file__).parent.joinpath("zg_timing.json").resolve()
assert zig_json_path.exists(), f"{zig_json_path!s} doesnt exist"
torch_json_path = Path(__file__).parent.joinpath("torch_timing.json").resolve()
assert torch_json_path.exists(), f"{torch_json_path!s} doesnt exist"

zig_train, zig_test = load_avg_trimmed_times(zig_json_path)
torch_train, torch_test = load_avg_trimmed_times(torch_json_path)

train_speedup = torch_train / zig_train
test_speedup = torch_test / zig_test

fig = go.Figure(
    data=[
        go.Bar(name="Train", x=["Speedup"], y=[train_speedup]),
        go.Bar(name="Test", x=["Speedup"], y=[test_speedup]),
    ]
)

fig.update_layout(
    title="Speedup of Zig over Torch Geometric (Avg Trimmed Epoch Times)",
    yaxis_title="Speedup (x)",
    barmode="group",
    xaxis_tickangle=-45,
)

fig.show()
print(f"{train_speedup}x train speed up (avg trimmed total epoch fbs times)")
print(f"{test_speedup}x train speed up (avg trimmed total epoch fbs times)")
