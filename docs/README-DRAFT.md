# Zigrad

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Zig](https://img.shields.io/badge/Zig-0.11.0-orange.svg)](https://ziglang.org/)

Zigrad is an experimental, high-performance machine learning framework written in Zig, designed with a focus on efficiency and minimal footprint.

## Overview

Zigrad aims to provide a lightweight yet powerful alternative to existing machine learning frameworks. By leveraging Zig's unique features, Zigrad offers:

- Competitive performance with established frameworks
- Minimal dependencies and small footprint
- Low-level control with high-level abstractions
- Automatic differentiation engine
- Support for both deep learning and reinforcement learning tasks

## Key Components

- `NDArray`: Efficient N-dimensional array operations with BLAS integration
- `NDTensor`: Extends `NDArray` with gradient tracking for automatic differentiation
- `GraphManager`: Manages the computational graph for efficient backpropagation
- Neural Network Layers: Linear, Convolutional (2D), Activation functions (ReLU, Softmax)
- Optimization: SGD and Adam optimizers
- Reinforcement Learning: Replay Buffer, DQN implementation

## Features

- Automatic differentiation
- Model serialization and deserialization
- Gradient clipping (by norm or value)
- Support for training and inference modes
- Integration with platform-specific BLAS libraries

## Performance Focus

Zigrad is designed with performance as a primary goal:

- Minimizes heap allocations
- Utilizes in-place operations where possible
- Plans for SIMD and GPU acceleration

## Current State and Roadmap

Zigrad is under active development. Key areas of focus include:

- Improving test data generation
- Architectural reorganization for better maintainability
- Implementing finite difference methods
- Developing a custom pool allocator
- Planning for operation fusion and SIMD optimizations
- Expanding reinforcement learning capabilities

Upcoming features:

- Multithreading support
- GPU acceleration
- Advanced network architectures (LSTM, Transformers)
- Lazy operation fusion

## Getting Started

### Prerequisites

- Zig 0.11.0 or later

### Building from Source

1. Clone the repository:
   ```
   git clone https://github.com/Marco-Christiani/zigrad.git
   cd zigrad
   ```

2. Build the project:
   ```
   zig build
   ```

## Usage Example

Here's a simple example of creating and training a DQN agent for the CartPole environment:

```zig
const zg = @import("zigrad");
const CartPole = @import("CartPole.zig");
const DQNAgent = @import("dqn_agent.zig").DQNAgent;

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.c_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var env = CartPole.init(zg.settings.seed);
    var agent = try DQNAgent(f32, 1000).init(allocator, .{
        .input_size = 4,
        .hidden_size = 128,
        .output_size = 2,
        .gamma = 0.99,
        .eps_start = 0.9,
        .eps_end = 0.05,
        .eps_decay = 1000,
        .optimizer = zg.optim.Adam(f32).init(allocator, 1e-3, 0.9, 0.999, 1e-8).optimizer(),
    });
    defer agent.deinit();

    // Training loop implementation...
}
```

## Architecture

Zigrad's architecture is evolving to support more complex operations and better maintainability:

- `NDArray` and `NDTensor` as core data structures
- Backend abstraction for different compute platforms (e.g., BLAS, CUDA)
- `Module` abstraction for handling trainable parameters
- Consideration of `Function` abstraction for custom operations

## Contributing

Contributions to Zigrad are welcome! Please check the roadmap for current focus areas and open issues for specific tasks.

## License

This project is protected under the [MIT License](https://opensource.org/licenses/MIT).

---

Zigrad is a personal project exploring the intersection of systems programming and machine learning. It serves as a learning platform and demonstrates Zig's capabilities in numerical computing and machine learning tasks.
