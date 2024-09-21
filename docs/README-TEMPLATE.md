[<img src="https://img.icons8.com/?size=512&id=55494&format=png" align="right" width="25%" padding-right="350">]()

# `Zigrad`

#### Headline

<p align="left">
	<img src="https://img.shields.io/github/license/Marco-Christiani/zigrad?style=flat&logo=opensourceinitiative&logoColor=white&color=0080ff" alt="license">
	<img src="https://img.shields.io/github/last-commit/Marco-Christiani/zigrad?style=flat&logo=git&logoColor=white&color=0080ff" alt="last-commit">
	<img src="https://img.shields.io/github/languages/top/Marco-Christiani/zigrad?style=flat&color=0080ff" alt="repo-top-language">
	<img src="https://img.shields.io/github/languages/count/Marco-Christiani/zigrad?style=flat&color=0080ff" alt="repo-language-count">
</p>
<p align="left">
	<img src="https://img.shields.io/badge/Zig-F7A41D.svg?style=flat&logo=Zig&logoColor=white" alt="Zig">
</p>

<br>

##### 🔗 Table of Contents

- [📍 Overview](#-overview)
- [👾 Features](#-features)
- [📂 Repository Structure](#-repository-structure)
- [🧩 Core Components](#-modules)
- [🚀 Getting Started](#-getting-started)
    - [🔖 Prerequisites](#-prerequisites)
    - [📦 Installation](#-installation)
    - [🤖 Usage](#-usage)
    - [🧪 Verification](#-tests)
- [📌 Project Roadmap](#-project-roadmap)
- [🤝 Contributing](#-contributing)
- [🎗 License](#-license)
- [🙌 Acknowledgments](#-acknowledgments)

---

## 📍 Overview


---

## 👾 Features

<!-- Rough ideas for a possible table with some attractive points -->

|    |   Feature         | Description |
|----|-------------------|---------------------------------------------------------------|
| 🔌 | **Design Principles**  ||
| ⚙️  | **Architecture**  ||
| 📄 | **Documentation** ||
| 🧩 | **Modularity**    ||
| 🧪 | **Verification**  | Verified against PyTorch. |
| ⚡️  | **Performance**   | X% faster on ... K ms inference on M sized network ...  |
| 📦 | **Dependencies**  | Zero dependencies. Optional platform-specific optimizations for Apple Silicon with Accelerate, Nvidia GPUs with CUDA, and cross-platform SIMD support. |

---

## 📂 Repository Structure

<!-- Rough sketch of what to show, only show the most important high level components, not the entire tree. -->

```sh
└── zigrad/
    ├── docs
    │   ├── architecture.norg
    │   └── roadmap.norg
    ├── src
    │   ├── backend/
    │   ├── ndarray/
    │   ├── scalar/
    │   ├── nn/
    │   ├── rl/
    │   ├── graph_manager.zig
    │   ├── ndarray.zig
    │   ├── ndtensor.zig
    │   ├── main.zig
    │   └── root.zig
    ├── build.zig
    └── justfile
```

---

## 🧩 Core Components


<details closed><summary>src</summary>

| Component | Summary |
| --- | --- |
| [`NDArray`](https://github.com/Marco-Christiani/zigrad/blob/main/src/ndarray.zig) |  |
| [`NDTensor`](https://github.com/Marco-Christiani/zigrad/blob/main/src/ndarray.zig) |  |
  | [`GraphManager`](https://github.com/Marco-Christiani/zigrad/blob/main/src/graph_manager.zig) |  |

</details>

<details closed><summary>src.backend</summary>

| Component | Summary |
| --- | --- |
| [backend](https://github.com/Marco-Christiani/zigrad/blob/main/src/backend/blas.zig) | `blas.zig` platform-specific linear algebra libraries for optimized numerical computations, dynamically selected based on the compilation target and abstracted away. |

</details>

<details closed><summary>src.nn</summary>

<!-- BEGIN dummy filler text -->
| File | Summary |
| --- | --- |
| [winit.zig](https://github.com/Marco-Christiani/zigrad/blob/main/src/nn/winit.zig) | Initializes neural network weights using the He initialization method. Calculates standard deviation based on input tensor shape, then assigns randomized values to the weights. |
| [optim.zig](https://github.com/Marco-Christiani/zigrad/blob/main/src/nn/optim.zig) | The `optim.zig` file within the `nn` module of the `zigrad` repository implements functions for clipping gradients and defining Optimizer behavior. The `clipGrads` function clips gradients based on specified options, enhancing stability during training. Additionally, the `Optimizer` struct defines methods for stepping through optimization iterations, contributing to the efficiency of the neural network training process. These functionalities are essential for optimizing model performance and enhancing training outcomes within the architecture of the parent repository. |
| [layer.zig](https://github.com/Marco-Christiani/zigrad/blob/main/src/nn/layer.zig) | The `layer.zig` file in the `nn` directory of the `zigrad` repository defines a customizable Layer struct. This struct encapsulates functionality for neural network layers, including methods for forward propagation and parameter initialization. It interfaces with essential components such as NDTensor, GraphManager, and optimization techniques like Stochastic Gradient Descent. By abstracting layer operations, this code promotes modularity and flexibility within the neural network architecture of the parent repository. |
| [loss.zig](https://github.com/Marco-Christiani/zigrad/blob/main/src/nn/loss.zig) | The loss.zig file in the nn module of the repository contributes a critical function for calculating Mean Squared Error (MSE) losses in a 1D setting within the neural network framework. Leveraging autograd capabilities, this function processes flat data by computing differences between predicted and actual values. By incorporating this functionality, the code enhances the model training process by enabling accurate error evaluation for improved neural network optimization. |
| [mlp.zig](https://github.com/Marco-Christiani/zigrad/blob/main/src/nn/mlp.zig) | The `MLP` structure in `mlp.zig` within the `src/nn` directory of the `zigrad` repository defines a multi-layer perceptron (MLP) model for neural network operations. It leverages components like layers (e.g., `LinearLayer`, `ReLULayer`) and optimization methods (e.g., Stochastic Gradient Descent) to support complex computations on tensors. This structure is central to implementing diverse neural network architectures for tasks such as training and inference within the project's computational graph framework.---This summary encapsulates the key functional aspects of the `MLP` structure in relation to the overarching architecture of the `zigrad` repository, emphasizing its role in neural network modeling and its integration with fundamental components for efficient computation and optimization. |
| [model.zig](https://github.com/Marco-Christiani/zigrad/blob/main/src/nn/model.zig) | Models neural network layers, allowing layer addition, forward propagation, parameter retrieval, gradient zeroing, and memory management in the Zig-based zigrad repositorys architecture. |
| [conv_test.zig](https://github.com/Marco-Christiani/zigrad/blob/main/src/nn/conv_test.zig) | The `conv_test.zig` file within the `src/nn` directory of the `zigrad` repository focuses on testing convolutional neural network functionalities. It utilizes modules for handling randomization, tensor operations, defining layers like Conv2D and ReLU, constructing models, managing computational graphs, implementing loss functions, and conducting model training. This code aids in ensuring the correctness and effectiveness of convolutional neural network components within the overarching machine learning architecture of the `zigrad` project. |
| [conv_utils.zig](https://github.com/Marco-Christiani/zigrad/blob/main/src/nn/conv_utils.zig) | The `conv_utils.zig` file in the `src/nn` directory of the `zigrad` repository contains a function named `im2col` that is crucial for implementing convolution operations in neural networks. This function transforms input data into a matrix format suitable for efficient convolution processing. By providing parameters such as kernel size, stride, padding, and dilation, this function streamlines the data preparation process for convolutions, enhancing the computational performance and effectiveness of neural network training within the repositorys architecture. |
| [utils.zig](https://github.com/Marco-Christiani/zigrad/blob/main/src/nn/utils.zig) | This code file in `src/nn/utils.zig` within the `zigrad` repository defines essential printing options for mathematical operations and tensors, enhancing clarity and customization. It leverages core functionalities from the parent architecture through the `zg` module, enabling seamless integration within the neural network components. The structured `PrintOptions` facilitate intuitive representation of operations and tensors, contributing to improved usability and interpretability across the framework. |
| [trainer.zig](https://github.com/Marco-Christiani/zigrad/blob/main/src/nn/trainer.zig) | The `trainer.zig` file in the `src/nn` directory of the `zigrad` repository defines a Trainer component responsible for training neural network models. It leverages various components from the parent architecture such as the GraphManager, NDTensor, Model, optimization algorithms like SGD, and loss functions including softmax cross-entropy and mean squared error. The Trainer struct allows for flexible selection of loss functions and plays a crucial role in training machine learning models within the larger framework. |
<!-- END dummy filler text -->

</details>

<details closed><summary>src.ndarray</summary>

<!-- BEGIN dummy filler text -->
| File | Summary |
| --- | --- |
| [Shape.zig](https://github.com/Marco-Christiani/zigrad/blob/main/src/ndarray/Shape.zig) | The Shape.zig file in the ndarray module of the repository defines a struct representing the shape of an ndarray. It provides a method to initialize the shape with specific dimensions and an allocator. This code ensures that the shape is valid and allocates memory accordingly, crucial for managing multi-dimensional arrays efficiently in the project's backend. |
| [utils.zig](https://github.com/Marco-Christiani/zigrad/blob/main/src/ndarray/utils.zig) | The `utils.zig` file in the `src/ndarray` directory of the `zigrad` repository provides essential utility functions for array manipulation. It includes functions to calculate the product of dimensions and determine the broadcasted shape for arrays. These utilities support the core array operations and optimizations within the broader architecture of the repository. |
<!-- END dummy filler text -->

</details>

<details closed><summary>src.scalar</summary>

The original proof of concept scalar implementation of autograd was scalar-based. Flexible but not scalable.

</details>

---

## 🚀 Getting Started

### 🔖 Prerequisites

**None**: `version x.y.z`

### 📦 Installation

Build the project from source:

1. Clone the zigrad repository:
```sh
❯ git clone https://github.com/Marco-Christiani/zigrad
```

2. Navigate to the project directory:
```sh
❯ cd zigrad
```

3. Install the required dependencies:
```sh
❯ ❯ INSERT-INSTALL-COMMANDS
```

### 🤖 Usage

#### MNIST Example

[mnist.zig](https://github.com/Marco-Christiani/zigrad/blob/main/src/nn/mnist.zig)

...

### 🧪 Verification

To verify against PyTorch...

```sh
❯ ❯ ...
```

---

## 📌 Project Roadmap

- [X] **`Task 1`**: <strike>Implement feature one.</strike>
- [ ] **`Task 2`**: Implement feature two.
- [ ] **`Task 3`**: Implement feature three.

---

## 🤝 Contributing

Contributions are welcome! Here are several ways you can contribute:

- **[Report Issues](https://github.com/Marco-Christiani/zigrad/issues)**: Submit bugs found or log feature requests for the `zigrad` project.
- **[Submit Pull Requests](https://github.com/Marco-Christiani/zigrad/blob/main/CONTRIBUTING.md)**: Review open PRs, and submit your own PRs.
- **[Join the Discussions](https://github.com/Marco-Christiani/zigrad/discussions)**: Share your insights, provide feedback, or ask questions.

<details closed>
<summary>Contributing Guidelines</summary>

1. **Fork the Repository**: Start by forking the project repository to your github account.
2. **Clone Locally**: Clone the forked repository to your local machine using a git client.
   ```sh
   git clone https://github.com/Marco-Christiani/zigrad
   ```
3. **Create a New Branch**: Always work on a new branch, giving it a descriptive name.
   ```sh
   git checkout -b new-feature-x
   ```
4. **Make Your Changes**: Develop and test your changes locally.
5. **Commit Your Changes**: Commit with a clear message describing your updates.
   ```sh
   git commit -m 'Implemented new feature x.'
   ```
6. **Push to github**: Push the changes to your forked repository.
   ```sh
   git push origin new-feature-x
   ```
7. **Submit a Pull Request**: Create a PR against the original project repository. Clearly describe the changes and their motivations.
8. **Review**: Once your PR is reviewed and approved, it will be merged into the main branch. Congratulations on your contribution!
</details>

<details closed>
<summary>Contributor Graph</summary>
<br>
<p align="left">
   <a href="https://github.com{/Marco-Christiani/zigrad/}graphs/contributors">
      <img src="https://contrib.rocks/image?repo=Marco-Christiani/zigrad">
   </a>
</p>
</details>

---

## 🎗 License

This project is protected under the [SELECT-A-LICENSE](https://choosealicense.com/licenses) License. For more details, refer to the [LICENSE](https://choosealicense.com/licenses/) file.

---

## 🙌 Acknowledgments

- List any resources, contributors, inspiration, etc. here.

---
