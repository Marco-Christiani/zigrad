# Basic deployment

This example owns one Zigrad PR program and packages it as a self-contained
IREE executable. The build has three products:

1. `emit-pr` constructs the program and writes its serialized PR artifact.
2. A host Zigrad compiler lowers that artifact to a target-specific VMFB.
3. `basic-deployment` includes the VMFB with `@embedFile` and invokes it through the
   target IREE runtime.

The deployed executable does not read model code or artifacts from the
filesystem. It uses fixed inputs and validates the output without formatting
it. The host output buffer has a fixed size and lives on the stack.

The maintained Nix packages are:

- `zigrad-example-basic-deployment-cpu`
- `zigrad-example-basic-deployment-cpu-aarch64`
- `zigrad-example-basic-deployment-cuda`, when the flake selects one CUDA architecture

These are repository validation profiles. The packaging function accepts other
Nix target package sets and IREE deployment profiles.

For example:

```sh
nix build .#zigrad-example-basic-deployment-cpu
./result/bin/basic-deployment
```
