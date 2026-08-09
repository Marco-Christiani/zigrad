# MNIST Training Example

This self-contained flake traces a three-layer model and a VJP-based SGD update,
 then runs the same training program through a selected compiler and runtime
 integration.

Create a new project from the repository template with:

```sh
nix flake init -t github:Marco-Christiani/zigrad#mnist
```

## Run

Enter the project development environment:

```sh
nix develop
zig build
zig build run -- --backend=pjrt --steps=1
zig build run -- --backend=iree --steps=1
```

PJRT is the default. Increase `--steps` for a longer run.

Build the packaged application with `nix build`.

A direct Zig workflow outside the development shell must provide the external
 SDK through `-Dsdk=<path>` or `ZG_EXTERNAL_SDK_ROOT`.

PJRT execution also requires an explicit `PJRT_PLUGIN_PATH`.

Use `zig build --fork=/path/to/zigrad` to test against a local Zigrad checkout
 without changing the pinned dependency.
