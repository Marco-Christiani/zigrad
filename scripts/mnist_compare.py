import re
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class PerformanceParser:
    def __init__(self):
        self.data = {"framework": [], "epoch": [], "batch": [], "loss": [], "ms_per_sample": []}

    def parse_zigrad_log(self, log_file):
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
                    self.data["framework"].append("Zigrad")
                    self.data["epoch"].append(current_epoch)
                    self.data["loss"].append(float(batch_match.group(1)))
                    self.data["batch"].append(int(batch_match.group(2)))
                    self.data["ms_per_sample"].append(float(batch_match.group(3)))
                    continue

    def parse_pytorch_csv(self, csv_file, batch_size):
        df = pd.read_csv(csv_file)
        loss_df = df[(df["type"] == "metric") & (df["name"] == "loss")]
        step_df = df[(df["type"] == "duration") & (df["name"] == "step")]

        for _, row in loss_df.iterrows():
            self.data["framework"].append("PyTorch")
            self.data["epoch"].append(row["epoch"])
            self.data["batch"].append(row["batch"])
            self.data["loss"].append(row["value"])

        for _, row in step_df.iterrows():
            # s -> ms -> ms/sample
            ms_per_sample = (row["value"] * 1000) / batch_size
            self.data["ms_per_sample"].append(ms_per_sample)

    def get_dataframe(self):
        return pd.DataFrame(self.data)

    def plot_results(self):
        df = self.get_dataframe()
        fig = make_subplots(rows=2, cols=1, subplot_titles=("Loss per Batch", "MS per Sample"))
        for framework in df["framework"].unique():
            framework_df = df[df["framework"] == framework]
            fig.add_trace(
                go.Scatter(
                    x=framework_df["epoch"] * len(framework_df["batch"].unique()) + framework_df["batch"],
                    y=framework_df["loss"],
                    mode="lines",
                    name=f"{framework} Loss",
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=framework_df["epoch"] * len(framework_df["batch"].unique()) + framework_df["batch"],
                    y=framework_df["ms_per_sample"],
                    mode="lines",
                    name=f"{framework} MS/Sample",
                ),
                row=2,
                col=1,
            )
        fig.update_layout(height=800, width=1000, title_text="Zigrad vs PyTorch Performance")
        fig.show()

    def print_summary(self):
        df = self.get_dataframe()
        for framework in df["framework"].unique():
            framework_df = df[df["framework"] == framework]
            print(f"\n--- {framework} Summary ---")
            print(f"Average loss: {framework_df['loss'].mean():.4f}")
            print(f"Average ms per sample: {framework_df['ms_per_sample'].mean():.4f}")


if __name__ == "__main__":
    parser = PerformanceParser()
    parser.parse_zigrad_log("/tmp/zigrad_log.txt")
    parser.parse_pytorch_csv(
        "/tmp/zg_torch_profile_data.csv", batch_size=64
    )  # Assuming batch_size=64, adjust if different
    parser.plot_results()
    parser.print_summary()
