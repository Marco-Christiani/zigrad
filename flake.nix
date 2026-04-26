{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

    flake-parts = {
      url = "github:hercules-ci/flake-parts";
      inputs.nixpkgs-lib.follows = "nixpkgs";
    };

    mirage-src = {
      url = "path:/home/marco/Github/mirage-c-api";
      flake = false;
    };

    mpk = {
      url = "path:/home/marco/flakes/mpk";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.mirage-src.follows = "mirage-src";
    };

    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs = {
        nixpkgs.follows = "nixpkgs";
        pyproject-nix.follows = "pyproject-nix";
      };
    };

    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs = {
        nixpkgs.follows = "nixpkgs";
        pyproject-nix.follows = "pyproject-nix";
        uv2nix.follows = "uv2nix";
      };
    };

    # Source pins (previously in nix/lock.json).
    # Update with: nix flake update --update-input xla-src (etc.)
    # LLVM and StableHLO commits are derived from XLA's third_party/;
    # after updating xla-src, check third_party/llvm/workspace.bzl
    # and third_party/stablehlo/workspace.bzl for new commits.
    xlaSrc = {
      url = "github:openxla/xla/913ae2eaa3cb88971003592a90959685a78c9e30";
      flake = false;
    };

    llvmSrc = {
      url = "github:llvm/llvm-project/8f264586d7521b0e305ca7bb78825aa3382ffef7";
      flake = false;
    };

    stablehloSrc = {
      url = "github:openxla/stablehlo/1ef9e390b5295e676d2b864fe1924bc2f3f4cf0f";
      flake = false;
    };

    # IREE compiler + its BYO-LLVM dependency.
    # ireeLlvmSrc: iree-org/llvm-project fork that IREE carries patches on top of.
    # ireeStablehloSrc: iree-org/stablehlo fork (diverges from openxla/stablehlo).
    # ireeFlatccSrc: flatcc library (IREE VM flatbuffer runtime).
    ireeSrc = {
      url = "github:iree-org/iree/776210bd36896f8ca14288637592a9d5cebfcea1";
      flake = false;
    };
    ireeLlvmSrc = {
      url = "github:iree-org/llvm-project/c95bd0bba5be9710292ce3a29832b67a0e33f887";
      flake = false;
    };
    ireeStablehloSrc = {
      url = "github:iree-org/stablehlo/6fabd27b15885179a3b6a601ea1e4171f2ed2c91";
      flake = false;
    };
    ireeFlatccSrc = {
      url = "github:dvidelabs/flatcc/9362cd00f0007d8cbee7bff86e90fb4b6b227ff3";
      flake = false;
    };
    ireeBenchmarkSrc = {
      url = "github:google/benchmark/192ef10025eb2c4cdd392bc502f0c852196baa48";
      flake = false;
    };
  };

  outputs = inputs @ {flake-parts, ...}: let
    # Project-wide CUDA configuration. Centralized to avoid hardcoding
    # sm_XX or toolkit versions in random places.
    cudaCfg = {
      gccHostAttr = "gcc14";
      cudaPackagesAttr = "cudaPackages_12_9";
      cudaVersion = "12.9.1";
    };

    # Build-time policy. Threaded into every long-running C++/bazel derivation.
    #  Defaults reflect a development build that maximizes runtime performance
    #  at the cost of binary portability. For redistributable artifacts,
    #  override withNativeTuning=false and cudaArchitectures=[]/[fat].
    #  See the Building section of the docs site for full knob documentation
    #  and concrete use-case recipes.
    buildCfg = {
      # ---- Debug / production switch -----------------------------------
      # When true: cmake RelWithDebInfo, bazel --copt=-g --strip=never, dontStrip.
      # Default false = production: Release, NDEBUG, stripped.
      withDebugSymbols = false;

      # ---- CPU codegen -------------------------------------------------
      # Emit non-portable native CPU instructions (-march=native -mavx2 -mfma).
      # Massive perf win on the build host, but binaries won't run on older or
      # different CPU families. False for distribution; true for dev/benchmarks.
      withNativeTuning = true;

      # ---- CUDA codegen ------------------------------------------------
      # GPU compute capabilities to compile for. Empty = upstream defaults
      # (XLA: fat sm_60..sm_90; TVM: cmake "all-major"). Pin to your dev GPU
      # for max perf + faster builds: e.g. ["86"] for Ampere consumer (RTX 30,
      # A10), ["89"] for Ada (RTX 40), ["90"] for Hopper (H100), or a list
      # ["80" "86" "89" "90"] for portable-fat.
      cudaArchitectures = [];

      # ---- LTO (opt-in) ------------------------------------------------
      # Link-time optimization. Adds ~30-50% to build time for a 5-10% runtime
      # gain on the inner-loop kernels. Build-system support varies; we plumb
      # the flag through and rely on bazel/cmake to honor it.
      enableLto = false;

      # ---- Escape hatches for experimentation --------------------------
      # Appended to CMAKE_CXX_FLAGS for cmake-driven builds.
      # Examples: ["-mllvm" "-polly"], ["-funroll-loops"], ["-fno-plt"].
      extraCxxFlags = [];

      # Appended to CMAKE_EXE_LINKER_FLAGS / CMAKE_SHARED_LINKER_FLAGS.
      # Examples: ["-Wl,--gc-sections"], ["-fuse-ld=mold"].
      extraLdFlags = [];

      # Appended to xla-pjrt's bazelBuildFlags.
      # Examples: ["--copt=-funroll-loops"], ["--config=monolithic"].
      extraBazelFlags = [];
    };
  in
    flake-parts.lib.mkFlake {inherit inputs;} {
      systems = [
        "x86_64-linux"
        "aarch64-linux"
      ];

      imports = [
        ./nix/flake/sdk.nix
        ./nix/flake/devshells.nix
        ./nix/flake/checks.nix
      ];

      perSystem = {system, ...}: let
        pkgs = import inputs.nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            # note to self: avoid enabling cudaSupport globally unless you need nixpkgs packages to flip CUDA paths.
            # it might have wide-reaching effects on unrelated packages.
            # cudaSupport = true;
          };
        };

        # Drift detection: cudaCfg pins two related-but-independent things.
        #  cudaVersion drives versions.json (cuda-redist tarballs from nvidia).
        #  cudaPackagesAttr selects a nixpkgs cuda set whose minor version
        #  floats with the nixpkgs lock. Assert their major.minor agree so a
        #  silent split (e.g. nixpkgs bumps to 13.x) fails loud at eval time.
        pkgsCudartVersion = pkgs.${cudaCfg.cudaPackagesAttr}.cuda_cudart.version;
        cudaVerMM = pkgs.lib.versions.majorMinor cudaCfg.cudaVersion;
        pkgsCudartVerMM = pkgs.lib.versions.majorMinor pkgsCudartVersion;
      in
        assert pkgs.lib.assertMsg
          (cudaVerMM == pkgsCudartVerMM)
          ''
            cudaCfg drift: cudaVersion=${cudaCfg.cudaVersion} (major.minor ${cudaVerMM})
              disagrees with nixpkgs ${cudaCfg.cudaPackagesAttr}.cuda_cudart.version=${pkgsCudartVersion} (major.minor ${pkgsCudartVerMM}).
              Update flake.nix cudaCfg or the nixpkgs lock.
          '';
        {
          _module.args = {
            inherit pkgs cudaCfg buildCfg;
          };
          formatter = pkgs.alejandra;
        };
    };
}
