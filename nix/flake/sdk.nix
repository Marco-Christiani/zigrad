# nix/flake/sdk.nix
#
# SDK assembly: leaf derivations, mkSdk compositor with feature flags,
# named presets, package exports, and apps.
{inputs, ...}: let
  zigradVersion = inputs.self.shortRev or inputs.self.dirtyShortRev or "dev";
in {
  perSystem = {
    pkgs,
    system,
    cudaCfg,
    ...
  }: let
    inherit (inputs) xlaSrc llvmSrc stablehloSrc;
    inherit (inputs) ireeSrc ireeLlvmSrc ireeStablehloSrc ireeFlatccSrc ireeBenchmarkSrc;
    inherit (pkgs) lib;

    cudaPackages = pkgs.${cudaCfg.cudaPackagesAttr};
    gccHost = pkgs.${cudaCfg.gccHostAttr};
    mirageRuntime =
      if builtins.hasAttr system inputs.mpk.packages
      then inputs.mpk.packages.${system}.mirage-runtime
      else null;

    zigradSrc = import ../source-filter.nix {
      inherit (pkgs) lib;
      root = ../..;
    };

    src = zigradSrc;

    inherit
      (import ../targets.nix {
        inherit pkgs cudaPackages gccHost src;
        inherit (cudaCfg) cudaArchitectures;
        zigradExternalSdk = sdkProfiles.full-gpu.full;
        inherit (pkgs) zig;
      })
      targets
      ;

    # -- Bazel deps hashes ---------------------------------------------
    # Configuration-specific; must be maintained per combination.
    xlaPjrtDepsHashes = {
      cpu-onednn-native = "sha256-vpI+i27sWrNS/qeICNav8lZJHcGsnx+C+e58oAyd3oE=";
      cuda-onednn-thunk-native = "sha256-ivbrLtStbE1IW9hTiqyL0KoBdK9IZRPXcDmokH01eCE=";
    };

    # -- Leaf derivations -----------------------------------------------

    # PJRT + XLA FFI headers (pure source copy, zero build cost).
    pjrtHeaders = pkgs.runCommand "pjrt-xla-ffi-headers" {} ''
      mkdir -p $out/include/xla/pjrt/c $out/include/xla/ffi/api
      cp ${xlaSrc}/xla/pjrt/c/*.h $out/include/xla/pjrt/c/
      cp ${xlaSrc}/xla/ffi/api/*.h $out/include/xla/ffi/api/
    '';

    # Compile-time CUDA headers used by Zig @cImport("nvrtc.h").
    # Keep this isolated from runtime CUDA DSOs, which come from PJRT runtime bundles.
    cudaCompileHeaders = pkgs.runCommand "cuda-compile-headers" {} ''
      mkdir -p "$out/include"
      cp -as ${cudaPackages.cudatoolkit}/include/. "$out/include/"
    '';

    # LLVM 22 built from XLA-pinned sources. Shared by SDK and TVM to ensure
    # they use the same LLVM version (same pass registry, no ABI conflicts).
    llvm = pkgs.callPackage ../llvm.nix {
      inherit xlaSrc llvmSrc;
    };

    xlaMlirStablehloCapiSdk = pkgs.callPackage ../xla-mlir-stablehlo-capi-sdk.nix {
      inherit xlaSrc stablehloSrc llvm;
    };

    mlirExtSrc = let
      fs = lib.fileset;
    in
      fs.toSource {
        root = ../../shim;
        fileset = fs.unions [
          ../../shim/CMakeLists.txt
          ../../shim/mlir_ext.cc
          ../../shim/zigrad
          ../../shim/test
        ];
      };

    zigradMlirExt = pkgs.callPackage ../zigrad-mlir-ext.nix {
      inherit xlaMlirStablehloCapiSdk llvm;
      src = mlirExtSrc;
    };

    # PJRT C API plugins from XLA (Bazel).
    xlaPjrtPlugins = pkgs.callPackage ../xla-pjrt-runtime-bazel.nix {
      inherit xlaSrc;
      cudaSupport = false;
      cpuMathLibrary = "onednn";
      cpuNativeTuning = true;
      depsHash = xlaPjrtDepsHashes.cpu-onednn-native;
    };

    xlaPjrtPluginsCuda = pkgs.callPackage ../xla-pjrt-runtime-bazel.nix {
      inherit xlaSrc;
      inherit (cudaCfg) cudaArchitectures cudaVersion;
      cudaSupport = true;
      cpuMathLibrary = "onednn-thunk";
      cpuNativeTuning = true;
      depsHash = xlaPjrtDepsHashes.cuda-onednn-thunk-native;
    };

    # CUDA redistributable bundle (pre-built NVIDIA DSOs from CDN).
    cudaRedist = pkgs.callPackage ../cuda-redist.nix {
      inherit (cudaCfg) cudaVersion;
    };

    # TVM with LLVM 22 (built from XLA-pinned sources).
    # Uses shared LLVM to match SDK, avoiding pass registry conflicts.
    tvmPkg = pkgs.callPackage ../tvm.nix {
      inherit cudaPackages gccHost llvm;
      inherit (cudaCfg) cudaArchitectures;
      cudaSupport = true;
    };

    tvmCpu = pkgs.callPackage ../tvm.nix {
      inherit cudaPackages gccHost llvm;
      inherit (cudaCfg) cudaArchitectures;
      cudaSupport = false;
    };

    # IREE compiler: BYO-LLVM path using iree-org/llvm-project fork.
    # ireeLlvm: LLVM+Clang+LLD+MLIR built from IREE's fork. Separate from our
    #   XLA-pinned llvm because the two forks diverge in MLIR internals.
    ireeLlvm = pkgs.callPackage ../iree-llvm.nix {
      inherit ireeLlvmSrc;
    };
    ireeCompiler = pkgs.callPackage ../iree-compiler.nix {
      inherit ireeSrc ireeStablehloSrc ireeFlatccSrc ireeBenchmarkSrc ireeLlvm;
    };

    # IREE runtime: combined libIREERuntime.so with CPU HAL drivers.
    ireeRuntime = pkgs.callPackage ../iree-runtime.nix {
      inherit ireeSrc ireeStablehloSrc ireeFlatccSrc ireeBenchmarkSrc ireeLlvm;
    };

    # -- mkSdk compositor -----------------------------------------------
    # Composites leaf derivations into { compile, runtime, full } bundles
    # driven by feature flags. Replaces the old manual symlinkJoin profiles.
    mkSdk = {
      mlir ? true,
      iree ? false,
      tvm ? true,
      gpu ? true,
      mirage ? false,
      mkl ? true,
    }: let
      hasMirage = mirage && mirageRuntime != null;
    in rec {
      compile = pkgs.symlinkJoin {
        name = "zigrad-sdk-compile";
        paths =
          [pjrtHeaders]
          ++ lib.optional mlir xlaMlirStablehloCapiSdk
          ++ lib.optional mlir zigradMlirExt
          ++ lib.optional tvm tvmPkg.dev
          ++ lib.optional gpu cudaCompileHeaders
          ++ lib.optional iree ireeCompiler
          ++ lib.optional iree ireeRuntime
          ++ lib.optional hasMirage mirageRuntime
          ++ lib.optional mkl pkgs.mkl;
      };

      runtime = pkgs.symlinkJoin {
        name = "zigrad-sdk-runtime";
        paths =
          [
            (
              if gpu
              then xlaPjrtPluginsCuda
              else xlaPjrtPlugins
            )
          ]
          ++ lib.optional gpu cudaRedist
          ++ lib.optional tvm tvmPkg
          ++ lib.optional hasMirage mirageRuntime;
      };

      full = pkgs.symlinkJoin {
        name = "zigrad-sdk";
        paths = [compile runtime];
      };
    };

    # -- Named presets ---------------------------------------------------
    sdkProfiles = {
      full-gpu = mkSdk {
        mlir = true;
        tvm = true;
        gpu = true;
      };
      mlir-cpu = mkSdk {
        mlir = true;
        tvm = false;
        gpu = false;
      };
      iree-cpu = mkSdk {
        mlir = false;
        iree = true;
        tvm = false;
        gpu = false;
      };
      tvm-gpu = mkSdk {
        mlir = false;
        tvm = true;
        gpu = true;
      };
      minimal = mkSdk {
        mlir = false;
        tvm = false;
        gpu = false;
      };
    };

    # -- Binary derivation -----------------------------------------------
    zigrad = pkgs.callPackage ../zigrad.nix {
      inherit zigradSrc;
      version = zigradVersion;
      sdk = sdkProfiles.full-gpu.compile;
      optimize = "ReleaseFast";
    };

    hostCheckTvmRuntimeFullCompiler = pkgs.writeShellScriptBin "zigrad-check-tvm-runtime-full-compiler" ''
      set -euo pipefail

      runtime_root="${sdkProfiles.full-gpu.runtime}"
      export LD_LIBRARY_PATH="$runtime_root/lib:$runtime_root/runtime/sys/lib:$runtime_root/runtime/nvidia/nvrtc/lib:$runtime_root/runtime/nvidia/nvjitlink/lib:/run/opengl-driver/lib:''${LD_LIBRARY_PATH:-}"

      exec ${zigrad}/bin/zigrad tvm-check-compiler-load "$@"
    '';
  in {
    packages =
      {
        zigrad = zigrad;
        zigrad-check-tvm-runtime-full-compiler = hostCheckTvmRuntimeFullCompiler;
      }
      // (lib.optionalAttrs (mirageRuntime != null) {
        mirage-runtime = mirageRuntime;
      })
      # SDK presets (generated from sdkProfiles)
      // lib.concatMapAttrs (name: profile: {
        "zigrad-sdk-${name}-compile" = profile.compile;
        "zigrad-sdk-${name}-runtime" = profile.runtime;
        "zigrad-sdk-${name}" = profile.full;
      })
      sdkProfiles
      // {
        # Individual components (for targeted builds / debugging)
        llvm = llvm;
        tvm = tvmPkg;
        tvm-dev = tvmPkg.dev;
        tvm-cpu = tvmCpu;
        cuda-redist = cudaRedist;
        xla-mlir-stablehlo-capi-sdk = xlaMlirStablehloCapiSdk;
        xla-pjrt-plugins = xlaPjrtPlugins;
        xla-pjrt-plugins-cuda = xlaPjrtPluginsCuda;
        iree-llvm = ireeLlvm;
        iree-compiler = ireeCompiler;
        iree-runtime = ireeRuntime;
        pjrt-headers = pjrtHeaders;

        # Editor / tooling
        gen-clangd = targets.editor.clangd;
        gen-nvim = targets.editor.nvim;
        m4 = targets.zigrad-m4.build;
      };

    apps =
      (lib.optionalAttrs (targets ? example-cuda) {
        example-cuda-target = {
          type = "app";
          program = "${targets.example-cuda.run}/bin/example-cuda";
        };
      })
      // {
        m4 = {
          type = "app";
          program = "${targets.zigrad-m4.run}/bin/zigrad-m4";
        };

        gen-clangd = {
          type = "app";
          program = "${targets.editor.clangd}/bin/gen-clangd";
        };

        gen-nvim = {
          type = "app";
          program = "${targets.editor.nvim}/bin/gen-nvim";
        };

        ccache = {
          type = "app";
          program = "${pkgs.ccache}/bin/ccache";
        };

        tvm-runtime-full-compiler-host-check = {
          type = "app";
          program = "${hostCheckTvmRuntimeFullCompiler}/bin/zigrad-check-tvm-runtime-full-compiler";
        };
      };
  };
}
