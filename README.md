<p align="center">
  <img src="./website/nuxt-content/public/api/zg-logo.svg" width=350>
</p>

<p align="center">
 <img src="https://img.shields.io/github/license/Marco-Christiani/zigrad?style=flat&logo=opensourceinitiative" alt="license">
 <img src="https://img.shields.io/github/last-commit/Marco-Christiani/zigrad?style=flat&logo=git&logoColor=white" alt="last-commit">
 <img src="https://img.shields.io/github/languages/top/Marco-Christiani/zigrad?style=flat&color=F7A41D" alt="repo-top-language">
 <img alt="Discord" src="https://img.shields.io/discord/1325584101809324113?style=flat">
</p>
<br>
<p align="center"><strong><i>Supporting AI innovation from ideation to results.</i></strong></p>

---

> 🚧 **Zigrad is under active development.**
> This is the `modular` rewrite - a ground-up redesign as a compiler-oriented ML framework. APIs and architecture are evolving rapidly.

---

Zigrad is a deep learning and ML compiler framework. Rather than tying you to a single runtime, Zigrad lowers your program to [MLIR](https://mlir.llvm.org/) and dispatches through pluggable backends and kernel providers, giving you a clear path from high-level model definition to hardware-optimized execution.

<p align="center">
  <img src="./website/nuxt-content/app/components/zigrad-pipeline.svg" width=700>
</p>

## Features

- **Forward and reverse-mode AD** - VJP and JVP support (check out the llama training demo!)
- **OpenXLA backend** - CPU/GPU/TPU/etc execution via XLA's portable runtime interface, dynamically loadable so no rebuild required to switch devices
- **IREE backend** - AOT compilation with IREE for edge deployment
- **Pluggable kernel providers** - [TVM](https://tvm.apache.org/) (MetaSchedule tuning) and [Mirage](https://github.com/mirage-project/mirage) Superoptimizer for custom kernel generation.
- **MLIR lowering** - clean MLIR lowering pipeline with dump support at every stage and strong [OpenXLA](https://openxla.org/) and [IREE](https://iree.dev/) support via [StableHLO](https://github.com/openxla/stablehlo).
<!-- - **Nix hermetic infrastructure** - hermetic devshell with LLVM, XLA, PJRT, TVM, and StableHLO pinned -->
<!-- ([StableHLO](https://github.com/openxla/stablehlo) dialect for OpenXLA backend)  -->

## Getting Started

> 🚧 **Not user friendly yet**
> Actively working on a proper on-ramp for users, this is not a primary concern at the moment.
> If you have NixOS (or at least Nix), you are in good shape, but know that building these dependencies is rather expensive the first time.
> On a 64 core threadripper with 64 GB RAM the longest build is XLA which takes around 30-40 minutes for perspective.
>
> Please check the "roadmap" below user-facing concerns.

Nix is effectively required at present. The build depends on an external SDK (`ZG_EXTERNAL_SDK_ROOT`) and many demos require a PJRT plugin (`PJRT_PLUGIN_PATH`) or equivalent depending on the backend. All dependencies for all scenarios are provided automatically inside the devshell. Outside of it you'll need to set these manually and have compatible versions of the SDK and plugin.

```sh
# Use either
# 1. direnv
direnv allow

# Or
# 2. directly
nix develop
```

### Build and Test

```sh
zig build -Doptimize=ReleaseFast -Dsdk=$ZG_EXTERNAL_SDK_ROOT -Dinstall-runtime-link=true

zig build -Dsdk=$ZG_EXTERNAL_SDK_ROOT test
```

### Run a Demo

A few basic XLA Demos

```sh
# Reverse-mode AD
./zig-out/bin/zigrad vjp-demo

# Training
./zig-out/bin/zigrad train-demo

# Dump intermediate representations
./zig-out/bin/zigrad --dump-optimized=/tmp/train-demo/dump.hlo --dump-pr=/tmp/train-demo/dump.zxpr --dump-mlir=/tmp/train-demo/dump.mlir train-demo
```

Train LLAMA with XLA

```sh
./zig-out/bin/zigrad llama-ft-demo-pr
```

## Architecture

The pipeline has (generally) four core stages:

| Stage | Name | Ownership | Role |
|------:|------|-----------|------|
| 1 | User Program | User | Define computation with Zigrad frontend API |
| 2 | PR | Zigrad | Transformations (AD, legality, annotations) |
| 3 | IM (i.e., StableHLO) | Shared | Serialized artifact handed to the backend |
| 4 | Backend | Mixed | Compile IM -> EA, execute EA -> results |

**Backends are runtime-loadable.** PJRT plugins, for example, are loaded at startup via `PJRT_PLUGIN_PATH`. Switching from CPU to GPU requires no recompilation.

**Kernel providers are optional.** A provider (TVM, Mirage) can claim PR subgraphs and produce compiled kernel artifacts invoked at execution time. The baseline lowering path always exists and is always correct.

See [`docs/DESIGN.md`](docs/DESIGN.md) for the full architecture specification.

## Project Layout

```
src/
  pr/         Program Representation, AD, op registry
  frontend/   High-level program builders
  lower/      PR -> StableHLO/MLIR lowering
  backend/    PJRT and IREE backends
  pipeline/   Pass infrastructure
  c/          All external C API bindings (MLIR, PJRT, TVM, Mirage)
  tvm/        TVM kernel provider (tuning, dispatch, kernel provider)
  mirage/     Mirage kernel provider
```

## Roadmap

- User-facing frontend APIs: our APIs are far too low level at the moment, this is actually intentional while the infrastructure evolves, but we plan a much simpler easier to use API shortly.
- Distributable artifacts: removing the Nix requirement for users who simply want to use the library, this is already possible but not documented

## Contributing

- [Join the Discord](https://discord.gg/JWSSfWj3Uf) and head to the dev channels
- Open an issue before starting a PR
