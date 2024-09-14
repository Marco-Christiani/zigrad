import json
import torch
import torch.nn as nn


def load_zigrad_model(json_file):
    with open(json_file, "r") as f:
        model_data = json.load(f)

    class ZigradModel(nn.Module):
        def __init__(self):
            super(ZigradModel, self).__init__()
            self.layers = nn.ModuleList()

            for layer_data in model_data["layers"]:
                if "linear" in layer_data:
                    linear_data = layer_data["linear"]
                    weights = torch.tensor(linear_data["weights"]["data"]).reshape(
                        linear_data["weights"]["shape"]
                    )
                    bias = torch.tensor(linear_data["bias"]["data"])

                    linear_layer = nn.Linear(
                        weights.shape[0], weights.shape[1], bias=True
                    )
                    linear_layer.weight.data = weights
                    linear_layer.bias.data = bias

                    self.layers.append(linear_layer)
                elif "relu" in layer_data:
                    self.layers.append(nn.ReLU())
                # only these two for now

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    model = ZigradModel()
    return model


model = load_zigrad_model("model_weights.json")

print(model)

input_tensor = torch.tensor([[1.2, -0.4]]).reshape(1, 2)
output = model(input_tensor)
print(f"Input shape: {input_tensor.shape}")
print(f"Output shape: {output.shape}")
print(f"Output: {output}")

# torch.save(model.state_dict(), "pytorch_model.pth")
# print("Model saved in PyTorch format")
