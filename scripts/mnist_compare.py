import re
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class PerformanceParser:
    def __init__(self):
        self.data = {"framework": [], "epoch": [], "batch": [], "loss": [], "ms_per_sample": []}

    def parse_log(self, log_file, framework):
        epoch_pattern = re.compile(r"Epoch (\d+): Avg Loss = ([\d.]+)")
        batch_pattern = re.compile(r"train_loss: ([\d.]+) \[(\d+)/\d+\] \[ms/sample: ([\d.]+)\]")
        current_epoch = 0
        with open(log_file, "r") as f:
            for line in f:
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
                    self.data["ms_per_sample"].append(float(batch_match.group(3)))
                    continue

    def get_dataframe(self):
        return pd.DataFrame(self.data)

    def plot_results(self):
        df = self.get_dataframe()
        fig = make_subplots(rows=2, cols=1, subplot_titles=("Loss per Batch", "MS per Sample"))

        for framework in df["framework"].unique():
            framework_df = df[df["framework"] == framework]
            loss_fig = go.Figure()
            ms_per_sample_fig = go.Figure()

            loss_trace = go.Scatter(
                x=framework_df["epoch"] * framework_df["batch"].max() + framework_df["batch"],
                y=framework_df["loss"],
                mode="lines",
                name=f"{framework} Loss",
            )
            ms_per_sample_trace = go.Scatter(
                x=framework_df["epoch"] * framework_df["batch"].max() + framework_df["batch"],
                y=framework_df["ms_per_sample"],
                mode="lines",
                name=f"{framework} MS/Sample",
            )

            # add traces to individual figures
            loss_fig.add_trace(loss_trace)
            ms_per_sample_fig.add_trace(ms_per_sample_trace)

            # save individual plots
            loss_fig.write_html(f"/tmp/{framework}_loss.html")
            ms_per_sample_fig.write_html(f"/tmp/{framework}_ms_per_sample.html")

            fig.add_trace(loss_trace, row=1, col=1)
            fig.add_trace(ms_per_sample_trace, row=2, col=1)

        fig.update_layout(height=800, width=1000, title_text="Zigrad vs PyTorch Performance")
        fig.show()

        fig.write_html("/tmp/zg_mnist_zg_torch_perf.html")

    def print_summary(self):
        df = self.get_dataframe()
        for framework in df["framework"].unique():
            framework_df = df[df["framework"] == framework]
            print(f"\n--- {framework} Summary ---")
            print(f"Average loss: {framework_df['loss'].mean():.4f}")
            print(f"Average ms per sample: {framework_df['ms_per_sample'].mean():.4f}")


if __name__ == "__main__":
    parser = PerformanceParser()
    parser.parse_log("/tmp/zg_mnist_log.txt", "Zigrad")
    parser.parse_log("/tmp/zg_mnist_torch_log.txt", "PyTorch")
    parser.plot_results()
    parser.print_summary()
