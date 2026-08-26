# LLaMA training

This example loads a LLaMA 3.2 1B SafeTensors checkpoint, traces either a
training step or inference loss, lowers the same PR program to StableHLO, and
executes it through PJRT or IREE.

From the Zigrad repository, run the CUDA package:

```sh
nix run --impure .#zigrad-example-llama-training -- --backend=pjrt --mode=training
nix run --impure .#zigrad-example-llama-training -- --backend=iree --mode=inference
```

The default checkpoint path is
`./weights/llama-3.2-1b-instruct/model.safetensors`. Set
`ZG_LLAMA_SAFETENSORS_PATH` or pass `--weights=PATH` to select another file.
PJRT also requires `PJRT_PLUGIN_PATH`. IREE reads the `ZG_IREE_*` environment
variables documented by its integration configuration.

Run the app with `--help` for shape, dtype, and iteration options. Inside a
development shell, `zig build` from this directory performs a compile-only
check against `ZG_EXTERNAL_SDK_ROOT`.
