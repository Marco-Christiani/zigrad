from pathlib import Path
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

        metadata["platform"] = f"{platform.system().lower()}-{platform.release().lower()}"
        metadata["python"] = platform.python_version()
        self.metadata[framework] = metadata

    @property
    def df(self):
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

    def create_speedup_figure(self):
        torch_ms = self.df["ms_per_sample"][self.df["framework"].str.lower() == "pytorch"]
        zg_ms = self.df["ms_per_sample"][self.df["framework"].str.lower() == "zigrad"]
        zg_speedup = torch_ms.to_numpy() / zg_ms.to_numpy()

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

        fig.update_layout(title="Speedup Zigrad/PyTorch")
        return fig

    def create_ms_per_sample_distribution(self):
        fig = go.Figure()
        for framework in ("zigrad", "pytorch"):
            framework_df = self.df[self.df["framework"].str.lower() == framework]
            fig.add_trace(go.Histogram(x=framework_df["ms_per_sample"], name=f"{framework} ms/sample"))
        fig.update_layout(title="Ms/Sample Distribution")
        return fig

    def create_ms_per_sample_scatter(self):
        fig = go.Figure()
        for framework in ("zigrad", "pytorch"):
            framework_df = self.df[self.df["framework"].str.lower() == framework]
            x = framework_df["epoch"] * framework_df["batch"].max() + framework_df["batch"]
            y = framework_df["ms_per_sample"]
            fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=f"{framework} ms/sample"))
        fig.update_layout(title="Ms/Sample")
        return fig

    def create_loss_scatter(self):
        fig = go.Figure()
        for framework in ("zigrad", "pytorch"):
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
                title = fig.layout.title.text.replace("/", "_").replace(" ", "").lower()
                # fig.write_html(f"/tmp/zg_mnist_zg_torch_perf_{i}_{title}{theme_short}.html")
                svg_path = Path(f"/tmp/zg_mnist_zg_torch_perf_{i}_{title}{theme_short}.svg")
                fig.write_image(svg_path)
                s = 'width="700" height="500" '
                svg_txt = svg_path.read_text().replace(s, "", 1)
                svg_path.write_text(svg_txt)

        combined_fig = make_subplots(
            rows=4, cols=1, vertical_spacing=0.08, subplot_titles=[fig.layout.title.text for fig in figures]
        )
        for i, fig in enumerate(figures, start=1):
            for trace in fig.data:
                combined_fig.add_trace(trace, row=i, col=1)

        metadata_text = self._format_metadata(self.metadata["PyTorch"])
        for i in range(4):
            combined_fig.layout.annotations[i].text += f"<br><sub>{metadata_text}</sub>"

        combined_fig.update_layout(
            template=theme,
            height=1400,
            width=1000,
            title_text="Zigrad vs PyTorch Performance<br><sub>MNIST Train Model Arch - Simple</sub>",
            showlegend=False,
        )

        combined_fig.write_html(f"/tmp/zg_mnist_zg_torch_perf{theme_short}.html")
        combined_fig.write_image(f"/tmp/zg_mnist_zg_torch_perf{theme_short}.svg")

        return combined_fig

    def _format_metadata(self, metadata):
        return ", ".join([f"{key}: {value}" for key, value in metadata.items()])

    def print_summary(self):
        df = self.df
        for framework in df["framework"].unique():
            framework_df = df[df["framework"] == framework]
            print(f"\n--- {framework} Summary ---")
            print(f"Metadata: {self._format_metadata(self.metadata.get(framework, {}))}")
            print(f"Avg loss: {framework_df['loss'].mean():.6f}")
            print(f"Std loss: {framework_df['loss'].std():.6f}")
            print(f"Avg ms/sample: {framework_df['ms_per_sample'].mean():.6f}")
            print(f"Std ms/sample: {framework_df['ms_per_sample'].std():.6f}")
        torch_ms_per_sample = df["ms_per_sample"][df["framework"].str.lower() == "pytorch"]
        zg_ms_per_sample = df["ms_per_sample"][df["framework"].str.lower() == "zigrad"]
        zg_speedup = torch_ms_per_sample.to_numpy() / zg_ms_per_sample.to_numpy()
        print("Speedup")
        print(pd.Series(zg_speedup).describe())


if __name__ == "__main__":
    parser = PerformanceParser()
    parser.parse_log("/tmp/zg_mnist_log.txt", "Zigrad")
    parser.parse_log("/tmp/zg_mnist_torch_log.txt", "PyTorch")
    _ = parser.plot_results(True)
    fig = parser.plot_results(True, "plotly_dark")
    parser.print_summary()
    # input()
    fig.show()
