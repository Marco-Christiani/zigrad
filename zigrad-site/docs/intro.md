---
sidebar_position: 1
---
## Getting Started

Only dependency is a BLAS library.

### Linux

On linux you have some options,

- MKL (recommended for best performance)
  - See https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html
  - Reccommend a system installation for simplicity although this can work with `conda` for example, just make sure you adjust the library paths as necessary.
- OpenBLAS
  - See https://github.com/OpenMathLib/OpenBLAS/wiki/Precompiled-installation-packages
  - Likely available through your package manager as `libopenblas-dev` or `openblas-devel`

### Apple

- Nothing :)

### Examples

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

