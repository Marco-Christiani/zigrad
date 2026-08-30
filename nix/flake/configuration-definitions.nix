let
  component = provider: target: {inherit provider target;};
in {
  zigrad = {
    description = "Zigrad core";
    pname = "zigrad";
  };

  "xla:cpu" = {
    compilers = [(component "xla" "cpu")];
    runtimes = [(component "pjrt" "cpu")];
    description = "Zigrad core + XLA CPU + PJRT CPU backends";
  };

  "xla:cuda" = {
    compilers = [(component "xla" "cuda")];
    runtimes = [(component "pjrt" "cuda")];
    description = "Zigrad core + XLA CUDA + PJRT CUDA backends";
  };

  "iree:cpu" = {
    compilers = [(component "iree" "cpu")];
    runtimes = [(component "iree" "cpu")];
    description = "Zigrad core + IREE CPU compiler + runtime";
  };

  "iree:cuda" = {
    compilers = [(component "iree" "cuda")];
    runtimes = [(component "iree" "cuda")];
    description = "Zigrad core + IREE CUDA compiler + runtime";
  };

  "xla:cpu+iree:cpu" = {
    compilers = [
      (component "xla" "cpu")
      (component "iree" "cpu")
    ];
    runtimes = [
      (component "pjrt" "cpu")
      (component "iree" "cpu")
    ];
    description = "Zigrad core + XLA CPU + PJRT CPU + IREE CPU compiler + runtime";
  };

  "xla:cuda+iree:cuda" = {
    compilers = [
      (component "xla" "cuda")
      (component "iree" "cuda")
    ];
    runtimes = [
      (component "pjrt" "cuda")
      (component "iree" "cuda")
    ];
    description = "Zigrad core + XLA CUDA + PJRT CUDA + IREE CUDA compiler + runtime";
  };

  "xla:cuda+iree:cpu" = {
    compilers = [
      (component "xla" "cuda")
      (component "iree" "cpu")
    ];
    runtimes = [
      (component "pjrt" "cuda")
      (component "iree" "cpu")
    ];
    description = "Zigrad core + XLA CUDA + PJRT CUDA + IREE CPU compiler + runtime";
  };

  "tvm:cpu" = {
    compilers = [(component "tvm" "cpu")];
    runtimes = [(component "tvm" "cpu")];
    description = "Zigrad core + TVM CPU compiler + runtime";
  };

  "tvm:cuda" = {
    compilers = [(component "tvm" "cuda")];
    runtimes = [(component "tvm" "cuda")];
    description = "Zigrad core + TVM CUDA compiler + runtime";
  };

  "xla:cpu+tvm:cpu" = {
    compilers = [(component "xla" "cpu")];
    kernelProviders = [(component "tvm" "cpu")];
    runtimes = [(component "pjrt" "cpu")];
    description = "Zigrad core + XLA CPU + PJRT CPU + TVM CPU kernel provider";
  };

  "xla:cuda+tvm:cuda" = {
    compilers = [(component "xla" "cuda")];
    kernelProviders = [(component "tvm" "cuda")];
    runtimes = [(component "pjrt" "cuda")];
    description = "Zigrad core + XLA CUDA + PJRT CUDA + TVM CUDA kernel provider";
  };

  "xla:cuda+mirage:cuda" = {
    compilers = [(component "xla" "cuda")];
    kernelProviders = [(component "mirage" "cuda")];
    runtimes = [(component "pjrt" "cuda")];
    description = "Zigrad core + XLA CUDA + PJRT CUDA + Mirage CUDA kernel provider";
  };

  iree-cpu-runtime = {
    runtimes = [(component "iree" "cpu")];
    description = "Zigrad core + IREE CPU runtime";
    expose = false;
  };

  iree-cuda-runtime = {
    runtimes = [(component "iree" "cuda")];
    description = "Zigrad core + IREE CUDA runtime";
    expose = false;
  };

  "xla:cuda+iree:cpu+tvm:cuda+mirage:cuda" = {
    compilers = [
      (component "xla" "cuda")
      (component "iree" "cpu")
    ];
    kernelProviders = [
      (component "tvm" "cuda")
      (component "mirage" "cuda")
    ];
    runtimes = [
      (component "pjrt" "cuda")
      (component "iree" "cpu")
    ];
    description = "Zigrad core + XLA CUDA + IREE CPU + TVM CUDA + Mirage CUDA";
  };

  dev-cuda-tvm-python = {
    compilers = [
      (component "xla" "cuda")
      (component "iree" "cpu")
    ];
    kernelProviders = [
      (component "tvm-python" "cuda")
      (component "mirage" "cuda")
    ];
    runtimes = [
      (component "pjrt" "cuda")
      (component "iree" "cpu")
    ];
    description = "Zigrad core + XLA CUDA + IREE CPU + Mirage CUDA + TVM CUDA Python bindings";
    expose = false;
  };

  example-benchmark = {
    compilers = [
      (component "xla" "cuda")
      (component "iree" "cpu")
    ];
    kernelProviders = [(component "tvm" "cuda")];
    runtimes = [
      (component "pjrt" "cuda")
      (component "iree" "cpu")
    ];
    description = "Zigrad benchmark + XLA CUDA + IREE CPU + TVM CUDA";
    expose = false;
    pname = "zigrad-example-benchmark-dependencies";
  };
}
