# CIFAR-10 lifecycle

This example uses one Zigrad model definition for GPU training and IREE AOT
inference. The model is a small NHWC residual convolutional classifier:

```text
32x32 RGB image
  -> 16-channel convolution
  -> 16-channel residual block
  -> stride-2, 32-channel convolution
  -> 32-channel residual block
  -> global average reduction
  -> 10-class linear classifier
```

Training differentiates `model.forward` through categorical cross-entropy and
applies SGD. It writes the resulting parameter tree as SafeTensors. Inference
traces the same `model.forward` function with a batch size of one, compiles the
PR program to an IREE VMFB, and executes it through the IREE runtime.

## Data

Download and extract the CIFAR-10 binary distribution from the
[dataset site](https://www.cs.toronto.edu/~kriz/cifar.html). Pass the directory
containing `data_batch_1.bin` through `data_batch_5.bin` and `test_batch.bin` to
the trainer.

## Train and evaluate

The Nix package supplies the CUDA PJRT plugin and its runtime environment:

```sh
nix build --cores 32 .#zigrad-example-cifar10-train
./result/bin/cifar10-train \
  --data=/path/to/cifar-10-batches-bin \
  --steps=5000 \
  --output=cifar10.safetensors
```

The trainer evaluates the final parameters on all 10,000 test records. Use
`--no-eval` for a compilation or execution smoke test.

## Package and run inference

These packages compile the inference PR and include deterministic initial
parameters so the complete AOT build remains hermetic:

```sh
nix build --cores 32 .#zigrad-example-cifar10-cpu
nix build --cores 32 .#zigrad-example-cifar10-cpu-aarch64
```

The native runner accepts one raw CIFAR-10 record. A second argument selects a
trained checkpoint; omitting it uses the packaged initial parameters.

```sh
./result/bin/cifar10-infer record.bin cifar10.safetensors
```

The AArch64 package uses the same interface and targets Cortex-A72 for the
Raspberry Pi 4.
