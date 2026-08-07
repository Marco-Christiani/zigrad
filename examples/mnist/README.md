# MNIST Training Example

This example traces a three-layer model and a VJP-based SGD update, compiles the
 training step with StableHLO and PJRT, then runs it on synthetic data.

## Run

Enter the Zigrad development environment from the repository root:

```sh
nix develop --impure
cd examples/mnist
zig build
PJRT_PLUGIN_PATH="$PJRT_CPU_PLUGIN_PATH" zig build run -- --steps=1
```

Increase `--steps` for a longer run. To use XLA's CUDA plugin on a host with a
 visible supported GPU, select `PJRT_GPU_PLUGIN_PATH` instead.

A direct Zig workflow outside the devshell must provide the external SDK through
 `-Dsdk=<path>` or `ZG_EXTERNAL_SDK_ROOT`, and set `PJRT_PLUGIN_PATH` explicitly.
