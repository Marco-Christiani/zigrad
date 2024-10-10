import platform
import re

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class PerformanceParser:
    def __init__(self):
        self.data = {"framework": [], "epoch": [], "batch": [], "loss": [], "ms_per_sample": []}
        self.metadata = {}

    def parse_log(self, log_file, framework):
        """Parse logs from zg or pytorch runs and extract performance-data."""
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

        metadata["Platform"] = f"{platform.system()} {platform.release()} ({platform.python_version()})"
        self.metadata[framework] = metadata

    def get_dataframe(self):
        df = pd.DataFrame(self.data)
        # Verify
        s = df.groupby("framework").size()
        assert len(s) == 2  # got logs from both
        assert s.iloc[0] == s.iloc[1]  # even split
        # discard slowest time to give pytorch an advantage (tracing) and keep torch from skewing the plot
        torch_times = df["ms_per_sample"][df.framework.str.lower() == "pytorch"]
        zg_times = df["ms_per_sample"][df.framework.str.lower() == "zigrad"]
        df = df.drop(index=[torch_times.idxmax(), zg_times.idxmax()])
        return df

    def plot_results(self, save_individual=False):
        df = self.get_dataframe()
        fig = make_subplots(
            rows=4,
            cols=1,
            subplot_titles=(
                "Loss per Batch",
                "Ms/Sample Distribution",
                "Ms/Sample",
                "Speedup Zigrad/PyTorch",
            ),
        )
        torch_ms_per_sample = df["ms_per_sample"][df["framework"].str.lower() == "pytorch"]
        zg_ms_per_sample = df["ms_per_sample"][df["framework"].str.lower() == "zigrad"]
        for framework in df["framework"].unique():
            framework_df = df[df["framework"] == framework]

            # create individual plots
            loss_scatter = go.Scatter(
                x=framework_df["epoch"] * framework_df["batch"].max() + framework_df["batch"],
                y=framework_df["loss"],
                mode="lines",
                name=f"{framework} loss",
            )

            ms_per_sample_hist = go.Histogram(
                x=framework_df["ms_per_sample"],
                name=f"{framework} ms/sample",
            )

            x = framework_df["epoch"] * framework_df["batch"].max() + framework_df["batch"]
            y = framework_df["ms_per_sample"]
            ms_per_sample_scatter = go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=f"{framework} ms/sample",
            )

            if save_individual:
                go.Figure(loss_scatter).write_html(f"/tmp/{framework.lower()}_loss.html")
                go.Figure(ms_per_sample_hist).write_html(f"/tmp/{framework.lower()}_ms_per_sample.html")
                go.Figure(ms_per_sample_scatter).write_html(f"/tmp/{framework.lower()}_ms_per_sample_scatter.html")

            fig.add_trace(loss_scatter, row=1, col=1)
            fig.add_trace(ms_per_sample_hist, row=2, col=1)
            fig.add_trace(ms_per_sample_scatter, row=3, col=1)

            # metadata subtitle
            metadata_text = self._format_metadata(self.metadata.get(framework, {}))
            fig.layout.annotations[0].text = f"loss/batch<br><sub>{metadata_text}</sub>"  # type: ignore
            fig.layout.annotations[1].text = f"ms/sample<br><sub>{metadata_text}</sub>"  # type: ignore
            fig.layout.annotations[2].text = f"ms/sample<br><sub>{metadata_text}</sub>"  # type: ignore

        zg_speedup = torch_ms_per_sample.to_numpy() / zg_ms_per_sample.to_numpy() - 1

        # The bars are hard to see if you have a lot of samples
        # fig.add_bar(
        #     y=zg_speedup,
        #     row=4,
        #     col=1,
        #     marker_color=["red" if x < 0 else "green" for x in zg_speedup],
        #     marker_cmid=0,
        #     marker_cauto=True,
        #     # marker_cmax=zg_speedup.max(),
        #     # marker_cmin=zg_speedup.min(),
        #     marker_colorscale="RdBu_r",
        #     # offset=0,
        #     # marker_width=0,
        # )
        speedup_range = zg_speedup.max() - zg_speedup.min()
        zero_offset_pct = abs(zg_speedup.min()) / speedup_range
        # eps_jitter = 0.01
        fig.add_scatter(
            y=zg_speedup,
            mode="lines",
            name=f"Speedup",
            fillgradient=dict(  # noqa: C408
                colorscale=[
                    (0, "red"),
                    (zero_offset_pct * abs(zg_speedup.min()) / 2, "red"),
                    # (zero_offset_pct * (1 - eps_jitter), "red"),
                    # (zero_offset_pct, "white"),
                    (zero_offset_pct * zg_speedup.max() / 2, "darkblue"),
                    (1, "blue"),
                ],
                type="vertical",
            ),
            fill="tozeroy",
            line_width=0,
            row=4,
            col=1,
        )
        fig.update_layout(
            template="plotly_dark",
            bargap=0,
            bargroupgap=0,
            height=1600,
            width=1000,
            title_text="Zigrad vs PyTorch Performance",
        )
        fig.show()

        fig.write_html("/tmp/zg_mnist_zg_torch_perf.html")

    def _format_metadata(self, metadata):
        return ", ".join([f"{key}: {value}" for key, value in metadata.items()])

    def print_summary(self):
        df = self.get_dataframe()
        for framework in df["framework"].unique():
            framework_df = df[df["framework"] == framework]
            print(f"\n--- {framework} Summary ---")
            print(f"Metadata: {self._format_metadata(self.metadata.get(framework, {}))}")
            print(f"Avg loss: {framework_df['loss'].mean():.4f}")
            print(f"Avg ms/sample: {framework_df['ms_per_sample'].mean():.4f}")


if __name__ == "__main__":
    parser = PerformanceParser()
    parser.parse_log("/tmp/zg_mnist_log.txt", "Zigrad")
    parser.parse_log("/tmp/zg_mnist_torch_log.txt", "PyTorch")
    parser.plot_results()
    parser.print_summary()
