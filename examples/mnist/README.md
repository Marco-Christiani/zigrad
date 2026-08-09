# MNIST Training Example

This example traces a three-layer model and a VJP-based SGD update, then runs the
 same training program through a selected compiler and runtime integration.

## Run

Enter the Zigrad development environment from the repository root:

```sh
nix develop
cd examples/mnist
zig build
PJRT_PLUGIN_PATH="$PJRT_CPU_PLUGIN_PATH" zig build run -- --backend=pjrt --steps=1
zig build run -- --backend=iree --steps=1
```

PJRT is the default. Increase `--steps` for a longer run. To use XLA's CUDA
 plugin on a host with a visible supported GPU, select `PJRT_GPU_PLUGIN_PATH`.

A direct Zig workflow outside the devshell must provide the external SDK through
 `-Dsdk=<path>` or `ZG_EXTERNAL_SDK_ROOT`. PJRT execution also requires an
 explicit `PJRT_PLUGIN_PATH`.
