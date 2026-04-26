{
  pkgs,
  system,
  pyproject-nix,
  uv2nix,
  pyproject-build-systems,
  cudaPackages,
  py ? pkgs.python312,
  py-pkgs ? pkgs.python312Packages,
}: let
  lib = pkgs.lib;

  workspace = uv2nix.lib.workspace.loadWorkspace {
    workspaceRoot = ./../..;
  };

  overlay = workspace.mkPyprojectOverlay {
    sourcePreference = "wheel";
  };

  hacks = pkgs.callPackage pyproject-nix.build.hacks {};

  hasPrefixAny = prefixes: target: lib.any (pre: lib.hasPrefix pre target) prefixes;
  pyCudaOverlay = final: prev: {
    jax = hacks.nixpkgsPrebuilt {
      from = py-pkgs.jax;
      prev = prev.jax;
    };
    jaxlib = hacks.nixpkgsPrebuilt {
      from = py-pkgs.jaxlib;
      prev = prev.jaxlib;
    };
    jax-cuda12-plugin = hacks.nixpkgsPrebuilt {
      from = py-pkgs.jax-cuda12-plugin;
      prev = prev.jax-cuda12-plugin;
    };
    jax-cuda12-pjrt = hacks.nixpkgsPrebuilt {
      from = py-pkgs.jax-cuda12-pjrt;
      prev = prev.jax-cuda12-pjrt;
    };

    torch = hacks.nixpkgsPrebuilt {
      # from = py-pkgs.torchWithoutCuda;
      # from = py-pkgs.torchWithCuda;
      from = py-pkgs.torch-bin;
      # from = py-pkgs.torch;
      prev = prev.torch.overrideAttrs (old: {
        passthru =
          old.passthru
          // {
            # dependencies = lib.filterAttrs (name: _: ! lib.hasPrefix "nvidia" name) old.passthru.dependencies;
            dependencies = lib.filterAttrs (name: _: ! (hasPrefixAny ["nvidia" "libnv" "libcu" "cudnn" "cuda" "nccl"] name)) old.passthru.dependencies;
          };
      });
    };
  };

  pythonSet =
    (pkgs.callPackage pyproject-nix.build.packages {
      python = py;
    }).overrideScope (
      lib.composeManyExtensions [
        pyproject-build-systems.overlays.default
        overlay
        pyCudaOverlay
      ]
    );

  pythonEnv =
    pythonSet.mkVirtualEnv
    "python-devshell-env"
    workspace.deps.default;
  cudaDeps = with cudaPackages; [
    cudatoolkit
    cudnn
    nccl
    pkgs.linuxPackages.nvidia_x11
    # libnvshmem
    # cutensor
    # libcutensor
    # cusparselt
    # libcublas
    # libcusparse
    # libcusolver
    # libcurand
    # cuda_gdb
    # cuda_nvcc
    cuda_cudart
    # libnvjitlink
  ];
  envpkgs =
    [
      pythonEnv
      pkgs.uv
      pkgs.git
    ]
    ++ cudaDeps;
in {
  out = {
    packages = envpkgs;

    env = {
      UV_PYTHON = pythonSet.python.interpreter;
      UV_PYTHON_DOWNLOADS = "never";
      UV_NO_SYNC = "1";
      # LD_LIBRARY_PATH = lib.makeLibraryPath (envpkgs ++ pkgs.pythonManylinuxPackages.manylinux1);
      # sensible default in case running something bare (w/o our wrappers)
      XLA_FLAGS = "--xla_gpu_cuda_data_dir=${cudaPackages.cudatoolkit}";
    };
  };
}
