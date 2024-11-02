<p align="center">
  <img src="./docs/zg-logo.svg" width=350>
</p>

<p align="center">
	<img src="https://img.shields.io/github/license/Marco-Christiani/zigrad?style=flat&logo=opensourceinitiative" alt="license">
	<img src="https://img.shields.io/github/last-commit/Marco-Christiani/zigrad?style=flat&logo=git&logoColor=white" alt="last-commit">
	<img src="https://img.shields.io/github/languages/count/Marco-Christiani/zigrad?style=flat" alt="repo-language-count">
	<img src="https://img.shields.io/github/languages/top/Marco-Christiani/zigrad?style=flat&color=F7A41D" alt="repo-top-language">
	<!-- <img src="https://img.shields.io/badge/Zig-F7A41D.svg?style=flat&logo=Zig&logoColor=white" alt="Zig"> -->
</p>
<br>

# Zigrad
#### A deep learning framework built on an autograd engine with high level abstractions and low level control.


https://github.com/user-attachments/assets/3842aa72-9b16-4c25-8789-eac7159e3768



**Fast**
<!-- benchmarks -->

2.5x+ speedup over a compiled PyTorch model on Apple Silicon, 1.5x on x86. Expect similar performance gains across more architectures and platforms as MKL/CUDA support improves and Zigrad's ML graph compiler is operational.*
<!-- link to a benchmarking page -->
<!-- only need one of the bm plots, probably fast vs fast since that requires the least explanation -->

<picture>
  <source media="(prefers-color-scheme: light)" srcset="docs/zg_mnist_zg_torch_perf.svg">
  <source media="(prefers-color-scheme: dark)" srcset="docs/zg_mnist_zg_torch_perf_dark.svg" >
  <img alt="Description of the image" src="docs/zg_mnist_zg_torch_perf.svg">
</picture>
<!-- ![](./docs/zg_mnist_zg_torch_perf_0_speedupzigrad_pytorch_plotly.svg) -->

<sub>*Tensorflow excluded for scaling purposes (too slow). A hermetic, reproducible benchmarking pipeline built on Bazel will allow testing across more platforms (in progress, testers needed).</sub>

**Built for specialized optimization**

Zigrad's design enables deep control and customization

- Fine-grained control over memory management
- Flexible tradeoffs between performance characteristics like latency vs throughput
- Optimize for your specific hardware, use case, and system requirements
- No abstraction layers or build systems that make aggressive optimizations challenging or complex

But wait, there's more..

- Tiny binaries: binaries for the MNIST tests shown are under 400kb in `ReleaseFast` mode and under 200kb in `ReleaseSmall`.
- Graph tracing
- Tensorboard integration*
- Cross platform
- Statically linked executables
- Minimal and transparent heap allocations
<!-- Scalar API -->

<sub>*Not yet merged</sub>

## Features

### Trace the Computation Graph

![](./docs/comp_graph_mnist_simple_noag.svg)

An example of tracing the computation graph generated by a fully connected neural network for MNIST.

- *Input:* Batch of images 28x28 pixel samples.
- **Flatten:** `28x28 -> 784`
- **FC1**: Linear layer `784 -> 128`
- **ReLU**
- **FC2:** Linear layer `128 -> 64`
- **ReLU**
- **FC3:** Linear layer `64 -> 10`
- *Output:* Value for each of the 10 classes


We did not have to use Zigrad's modules to write this network at all, as Zigrad is backed by a capable autograd engine. Even when using the autograd backend to dynamically construct the same neural network Zigrad can still trace the graph and render it.

  > Note: Since the graph is generated from the autograd information, we set the labels for the nodes by naming the tensors for the sake of the diagram.

![](./docs/comp_graph_mnist_simple_ag.svg)

## Getting Started

The `examples/` directory has some standalone templates you can take and modify, the zon files are pinned to commit hashes.

Hello world example shows how to run a backward pass using the `GraphManager.` Note that in this very simple example, we do not need the `GraphManager` and the script could be simplified but this is designed to get you familiar with the workflow.

```shell
git clone https://github.com/Marco-Christiani/zigrad/
cd zigrad/examples/hello-world
zig build run
```

Run the mnist demo

```shell
cd zigrad/examples/mnist
make help
make
```

## Roadmap

A lot is planned and hoping for support from the Zig community so we can accomplish some of the more ambitious goals.

- More comprehensive MKL support
- More parallelization (e.g. activation functions)
- CUDA support
- Lazy tensors
- Static graph optimization
- Dynamic graph compiler
- MLIR
- Support for popular formats like ONNX and ggml.
- ZML translation for inference

## Known Issues and Limitations

- Lack of GPU support for now
- Effort has been directed towards performant primitives, not many layer types have been implemented
  - e.g. conv, pooling, etc are test implementations for verification, they are slow and unoptimized, I would not use them
