# C++ MLIR Integration

This example implements a Zigrad dialect, StableHLO rewrite passes, an
`mlir-opt` plugin, and an MLIR language server in C++.

It demonstrates how a C++ or MLIR consumer can extend the current Zigrad
compilation path.

The resident `src/` implementation does not load this plugin. Its active
kernel-provider path selects and compiles PR regions in Zig.

Build and test the plugin package with:

```console
nix build .#zigrad-example-mlir-cpp
```

Enter the opt-in development shell with:

```console
nix develop .#mlir-cpp
```

The shell provides the example's `mlir-lsp-server` and the broad Zigrad
development inputs.

Configure a local build directory for clangd with:

```console
nix run .#configure-mlir-cpp
```

The scoped `.clangd` file reads the resulting compilation database from
`examples/mlir-cpp/build`. Editors can use `mlir-lsp-server` from the
`mlir-cpp` shell for `.mlir` files.
