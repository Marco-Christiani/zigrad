---
title: Getting Started
sidebar_position: 1
---
## Getting Started

### Linux

On linux you have some options for a BLAS library (if you do not already have one installed),

- MKL (recommended for best performance)
  - See https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html
  - Recommend a system installation for simplicity although this can work with `conda` for example, just make sure you adjust the library paths as necessary.
- OpenBLAS
  - See https://github.com/OpenMathLib/OpenBLAS/wiki/Precompiled-installation-packages
  - Likely available through your package manager as `libopenblas-dev` or `openblas-devel`

### Apple

- Nothing

### Examples

The `examples/` directory has some standalone templates you can take and modify. For stable examples the zon files are pinned to commit hashes (by contrast, some examples are on the bleeding edge of Zigrad).

#### Hello World

```shell
git clone https://github.com/Marco-Christiani/zigrad/
cd zigrad/examples/hello-world
zig build run
```


#### MNIST

Run the mnist demo

```shell
cd zigrad/examples/mnist
make help
make
```

