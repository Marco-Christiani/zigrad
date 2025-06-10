# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "kaleido",
#     "numpy",
#     "pandas",
#     "plotly",
#     "scipy",
# ]
# ///
"""Timing analysis of log files produced by logging redirect for minimal backpressure.


Example
=======

```sh
# Profile torch
uv run src/nn/tests/test_mnist.py -t --batch_size=64 --num_epochs=3 --model_variant=simple --autograd > /tmp/zg_mnist_torch_log.txt

# Profile zigrad
cd examples/mnist
zig build -Doptimize=ReleaseFast
./zig-out/bin/main 2> /tmp/zg_mnist_log.txt

# Analyze
uv run scripts/mnist_compare.py
```
"""

import platform
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

other_framework_name = "torch"
# other_framework_name = "jax"
# other_framework_name = "tf"


class PerformanceParser:
    def __init__(self):
        self.data = {"framework": [], "epoch": [], "batch": [], "loss": [], "ms_per_sample": []}
        self.metadata = {}

    def parse_log(self, log_file, framework):
        """Parse logs from zg or pytorch runs and extract performance-data."""
        print("Loading", log_file, "for", framework)
        batch_size_pattern = re.compile(r"batch_size=(\d+)")
        learning_rate_pattern = re.compile(r"learning_rate=([\d.]+)")
        num_epochs_pattern = re.compile(r"num_epochs=(\d+)")
        device_pattern = re.compile(r"device=(\w+)")
        n_batches_pattern = re.compile(r"n_batches=(\d+)")
        grad_mode_pattern = re.compile(r"grad_mode=(\w+)")

        epoch_pattern = re.compile(r"Epoch (\d+): Avg Loss = ([\d.]+)")
        batch_pattern = re.compile(r"train_loss: ([\d.]+) \[(\d+)/(\d+)\] \[ms/sample: ([\d.]+)\]")

        current_epoch = 0
        metadata = {}

        with open(log_file, "r") as f:
            for line in f:
                # extract metadata from the first few lines
                if batch_size_match := batch_size_pattern.search(line):
                    metadata["batch_size"] = batch_size_match.group(1)
                if num_epochs_match := num_epochs_pattern.search(line):
                    metadata["epochs"] = num_epochs_match.group(1)
                if learning_rate_match := learning_rate_pattern.search(line):
                    metadata["lr"] = learning_rate_match.group(1)
                if device_match := device_pattern.search(line):
                    metadata["device"] = device_match.group(1)
                if n_batches_match := n_batches_pattern.search(line):
                    metadata["batches"] = n_batches_match.group(1)
                if grad_mode_match := grad_mode_pattern.search(line):
                    metadata["pt_grad_mode"] = grad_mode_match.group(1)

                epoch_match = epoch_pattern.search(line)
                if epoch_match:
                    current_epoch = int(epoch_match.group(1))
                    continue
                batch_match = batch_pattern.search(line)
                if batch_match:
                    self.data["framework"].append(framework)
                    self.data["epoch"].append(current_epoch)
                    self.data["loss"].append(float(batch_match.group(1)))
                    self.data["batch"].append(int(batch_match.group(2)))
                    self.data["ms_per_sample"].append(float(batch_match.group(4)))

        metadata["platform"] = f"{platform.system().lower()}-{platform.release().lower()}"
        metadata["python"] = platform.python_version()
        self.metadata[framework] = metadata

    @lru_cache()
    def build_df(self, drop_slowest: bool = True) -> pd.DataFrame:
        timings_df = pd.DataFrame(self.data)
        # Verify
        grouped = timings_df.groupby("framework")
        s = grouped.size()
        assert len(s) == 2  # got logs from both

        torch_times = timings_df["ms_per_sample"][timings_df.framework.str.lower() == other_framework_name]
        zg_times = timings_df["ms_per_sample"][timings_df.framework.str.lower() == "zigrad"]

        if s.iloc[0] != s.iloc[1]:  # uneven split
            imbalance = abs(s.iloc[0] - s.iloc[1])
            etol = timings_df["epoch"].max().max()
            if imbalance <= etol:
                print("Did not get same number of batches for each framework but within epoch off-by-one tolerance", etol)
                print(s)
                min_size = grouped.size().min()
                timings_df = grouped.head(min_size).reset_index(drop=True)
                print("Trimming to", s.min(), "per framework")
            else:
                print(f"Number of batches for each framework differed significantly: {imbalance=}")
                print(s)
                raise ValueError("Parse or setup error in the logs")

        if drop_slowest:
            # discard slowest time to give pytorch an advantage (tracing) and keep torch from skewing the plot
            print("Dropping slowest to give torch an advantage")
            timings_df = timings_df.drop(index=[torch_times.idxmax(), zg_times.idxmax()])  # type: ignore

        return timings_df

    @property
    def df(self) -> pd.DataFrame:
        return self.build_df()

    def create_speedup_figure(self):
        torch_ms = self.df["ms_per_sample"][self.df["framework"].str.lower() == other_framework_name]
        zg_ms = self.df["ms_per_sample"][self.df["framework"].str.lower() == "zigrad"]
        zg_speedup = torch_ms.to_numpy() / zg_ms.to_numpy()  # type: ignore

        fig = go.Figure()
        fig.add_scatter(y=zg_speedup, mode="lines", name="Speed ratio")

        for y_value, label in [(1, "No Speedup (1x)"), (2, "2x Speedup"), (3, "3x Speedup")]:
            fig.add_hline(y=y_value, line_dash="dot", line_color="gray", opacity=0.8)
            fig.add_annotation(
                x=0,
                y=y_value + 0.1,
                text=label,
                xref="paper",
                yref="y",
                showarrow=False,
                font={"size": 12, "color": "black"},
                opacity=0.8,
            )

        fig.update_layout(title=f"Speedup Zigrad/{other_framework_name}")
        return fig

    def create_ms_per_sample_distribution(self):
        fig = go.Figure()
        for framework in ("zigrad", other_framework_name):
            framework_df = self.df[self.df["framework"].str.lower() == framework]
            fig.add_trace(go.Histogram(x=framework_df["ms_per_sample"], name=f"{framework} ms/sample"))
        fig.update_layout(title="Ms/Sample Distribution")
        return fig

    def create_ms_per_sample_scatter(self):
        fig = go.Figure()
        for framework in ("zigrad", other_framework_name):
            framework_df = self.df[self.df["framework"].str.lower() == framework]
            x = framework_df["epoch"] * framework_df["batch"].max() + framework_df["batch"]
            y = framework_df["ms_per_sample"]
            fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=f"{framework} ms/sample"))
        fig.update_layout(title="Ms/Sample")
        return fig

    def create_loss_scatter(self):
        fig = go.Figure()
        for framework in ("zigrad", other_framework_name):
            framework_df = self.df[self.df["framework"].str.lower() == framework]
            x = framework_df["epoch"] * framework_df["batch"].max() + framework_df["batch"]
            y = framework_df["loss"]
            fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=f"{framework} loss"))
        fig.update_layout(title="Loss per Batch")
        return fig

    def plot_results(self, save_individual=False, theme: str = "plotly"):
        margin = {"t": 150, "b": 5, "l": 5, "r": 5}
        figures = [
            self.create_speedup_figure(),
            self.create_ms_per_sample_distribution(),
            self.create_ms_per_sample_scatter(),
            self.create_loss_scatter(),
        ]

        theme_short = theme.replace("plotly", "")
        if save_individual:
            for i, fig in enumerate(figures):
                fig.update_layout(margin=margin, template=theme)
                title = fig.layout.title.text.replace("/", "_").replace(" ", "").lower()  # type: ignore
                # fig.write_html(f"/tmp/zg_mnist_zg_torch_perf_{i}_{title}{theme_short}.html")
                svg_path = Path(f"/tmp/zg_mnist_zg_{other_framework_name}_perf_{i}_{title}{theme_short}.svg")
                fig.write_image(svg_path)
                s = 'width="700" height="500" '
                svg_txt = svg_path.read_text().replace(s, "", 1)
                svg_path.write_text(svg_txt)

        combined_fig = make_subplots(
            rows=4,
            cols=1,
            vertical_spacing=0.08,
            subplot_titles=[fig.layout.title.text for fig in figures],  # type: ignore
        )
        for i, fig in enumerate(figures, start=1):
            for trace in fig.data:
                combined_fig.add_trace(trace, row=i, col=1)

        metadata_text = self._format_metadata(self.metadata[other_framework_name])
        for i in range(4):
            combined_fig.layout.annotations[i].text += f"<br><sub>{metadata_text}</sub>"  # type: ignore

        combined_fig.update_layout(
            template=theme,
            height=1400,
            width=1000,
            title_text=f"Zigrad vs {other_framework_name.capitalize()} Performance<br><sub>MNIST Train Model Arch - Simple</sub>",
            showlegend=False,
        )

        combined_fig.write_html(f"/tmp/zg_mnist_zg_{other_framework_name}_perf{theme_short}.html")
        combined_fig.write_image(f"/tmp/zg_mnist_zg_{other_framework_name}_perf{theme_short}.svg")

        return combined_fig

    def _format_metadata(self, metadata):
        return ", ".join([f"{key}: {value}" for key, value in metadata.items()])

    def print_summary(self, drop_slowest: bool):
        timings_df = self.build_df(drop_slowest=drop_slowest)
        for framework in timings_df["framework"].unique():
            framework_df = timings_df[timings_df["framework"] == framework]
            print(f"\n--- {framework} Summary ---")
            print(f"Metadata: {self._format_metadata(self.metadata.get(framework, {}))}")
            print(f"Avg loss: {framework_df['loss'].mean():.6f}")
            print(f"Std loss: {framework_df['loss'].std():.6f}")
            print(f"Avg ms/sample: {framework_df['ms_per_sample'].mean():.6f}")
            print(f"Std ms/sample: {framework_df['ms_per_sample'].std():.6f}")

        zg = timings_df[timings_df.framework.str.lower() == "zigrad"].reset_index(drop=True)
        pt = timings_df[timings_df.framework.str.lower() == other_framework_name].reset_index(drop=True)

        if drop_slowest:
            zg = zg.drop(index=[zg["ms_per_sample"].idxmax()]).reset_index(drop=True)  # type: ignore
            pt = pt.drop(index=[pt["ms_per_sample"].idxmax()]).reset_index(drop=True)  # type: ignore

        zg_ms = zg["ms_per_sample"]
        pt_ms = pt["ms_per_sample"]

        diffs = pt_ms - zg_ms
        speedup = pt_ms / zg_ms
        print("Speedup")
        print(speedup.describe())  # type: ignore

        # Stats tests

        t_stat, t_p = stats.ttest_rel(pt_ms, zg_ms)
        w_stat, w_p = stats.wilcoxon(diffs)

        # Effect sizes
        cohen_d = diffs.mean() / diffs.std(ddof=1)
        wins = (diffs > 0).sum()
        losses = (diffs < 0).sum()
        rank_biserial = (wins - losses) / len(diffs)

        # Bootstrap some CIs
        def bootstrap_ci(data, func, n_boot=5000, ci=95):
            boot_stats = [func(np.random.choice(data, size=len(data), replace=True)) for _ in range(n_boot)]
            lower, upper = np.percentile(boot_stats, [(100 - ci) / 2, 100 - (100 - ci) / 2])
            return lower, upper

        ci_mean_diff = bootstrap_ci(diffs.to_numpy(), np.mean)  # type: ignore
        ci_median_diff = bootstrap_ci(diffs.to_numpy(), np.median)  # type: ignore
        ci_speedup = bootstrap_ci(speedup, np.mean)

        print("\n--- Statistical Tests ---")
        print(f"Paired t-test: t = {t_stat:.3f}, p = {t_p:.3e}")
        print(f"Wilcoxon signed-rank: W = {w_stat:.3f}, p = {w_p:.3e}")

        print("\n--- Effect Sizes ---")
        print(f"Cohen's d: {cohen_d:.3f}")
        print(f"Rank-biserial correlation: {rank_biserial:.3f}")
        print(f"Zigrad wins in {wins / len(diffs):.2%} of batches")

        print("\n--- Bootstrap 95% Confidence Intervals ---")
        print(f"Mean latency diff (ms/sample): {ci_mean_diff[0]:.6f} - {ci_mean_diff[1]:.6f}")
        print(f"Median latency diff (ms/sample): {ci_median_diff[0]:.6f} - {ci_median_diff[1]:.6f}")
        print(f"Mean speedup (Zigrad/{other_framework_name.capitalize()}: {ci_speedup[0]:.2f}x - {ci_speedup[1]:.2f}x")


if __name__ == "__main__":
    print("Comparing against", other_framework_name)
    parser = PerformanceParser()
    parser.parse_log("/tmp/zg_mnist_log.txt", "Zigrad")
    parser.parse_log(f"/tmp/zg_mnist_{other_framework_name}_log.txt", other_framework_name)
    _ = parser.plot_results(True)
    fig = parser.plot_results(True, "plotly_dark")
    parser.print_summary(drop_slowest=True)
    fpath = f"/tmp/zg_mnist_{other_framework_name}_parsed.csv"
    print("Saving parsed results to", fpath)
    parser.df.to_csv(fpath)
    input()
    fig.show()
