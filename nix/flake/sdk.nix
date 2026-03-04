# nix/flake/sdk.nix
#
# SDK assembly: the LLVM → StableHLO → TVM → PJRT dependency graph,
# all package exports, and apps (which share the `targets` import).
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
        inherit
          pkgs
          cudaPackages
          gccHost
          src
          ;
        inherit (cudaCfg) cudaArchitectures;
        # FIXME: this is currently intentionally the impure devel variant
        zigradExternalSdk = zigradExternalSdkDevel;
        inherit (pkgs) zig;
      })
      targets
      ;

    # buildBazelPackage fetchAttrs hashes for xla-pjrt-runtime-bazel.nix.
    # These hashes are configuration-specific (CUDA on/off, cpuMathLibrary,
    # native tuning flags, etc) and must be maintained per combination.
    xlaPjrtDepsHashes = {
      cpu-onednn-native = "sha256-vpI+i27sWrNS/qeICNav8lZJHcGsnx+C+e58oAyd3oE=";
      # cuda-onednn-thunk-native = "sha256-EFE6NyvyOoereFfwwExFu03B6IjfjrBWD5ri6S3F/9Y=";
      cuda-onednn-thunk-native = "sha256-ECduu/VXD+wsTKex9lfo+B6zXZdvw1nMWWEE5D/FtY8=";
    };

    # Compile-time CUDA headers used by Zig @cImport("nvrtc.h").
    # Keep this isolated from runtime CUDA DSOs, which come from PJRT runtime bundles.
    cudaCompileHeaders = pkgs.runCommand "cuda-compile-headers" {} ''
      mkdir -p "$out/include"
      cp -as ${cudaPackages.cudatoolkit}/include/. "$out/include/"
    '';

    xlaMlirStablehloCapiSdk = pkgs.callPackage ../xla-mlir-stablehlo-capi-sdk.nix {
      inherit xlaSrc stablehloSrc llvm;
    };

    xlaMlirStablehloCapiDevel = pkgs.callPackage ../xla-mlir-stablehlo-capi-sdk.nix {
      inherit xlaSrc stablehloSrc llvm;
      stdenv = pkgs.ccacheStdenv;
      devel = true;
    };

    # LLVM 22 built from XLA-pinned sources. Shared by SDK and TVM to ensure
    # they use the same LLVM version (same pass registry, no ABI conflicts).
    llvm = pkgs.callPackage ../llvm.nix {
      inherit xlaSrc llvmSrc;
    };

    # SDK profiles:
    # - build: compile-time headers/libs (TVM + CUDA headers, no TVM runtime DSOs)
    # - runtime-full: includes TVM runtime/compiler DSOs
    # - runtime-no-tvm: intentionally excludes TVM runtime/compiler DSOs
    # - aggregate: build + runtime-full for default ergonomic workflows
    zigradExternalSdkBuild = pkgs.symlinkJoin {
      name = "zigrad-external-sdk-build";
      paths =
        [
          xlaMlirStablehloCapiSdk
          zigradMlirExt
          tvm.dev
          cudaCompileHeaders
          pkgs.mkl
        ]
        ++ pkgs.lib.optionals (mirageRuntime != null) [
          mirageRuntime
        ];
    };

    zigradExternalSdkRuntimeFull = pkgs.symlinkJoin {
      name = "zigrad-external-sdk-runtime-full";
      paths = [
        xlaPjrtPluginsCuda
        tvm
      ];
    };

    zigradExternalSdkRuntimeNoTvm = pkgs.symlinkJoin {
      name = "zigrad-external-sdk-runtime-no-tvm";
      paths = [
        xlaPjrtPluginsCuda
      ];
    };

    # Convenience aggregate.
    zigradExternalSdk = pkgs.symlinkJoin {
      name = "zigrad-external-sdk";
      paths = [
        zigradExternalSdkBuild
        zigradExternalSdkRuntimeFull
      ];
    };

    zigradExternalSdkBuildDevel = pkgs.symlinkJoin {
      name = "zigrad-external-sdk-build-devel";
      paths =
        [
          xlaMlirStablehloCapiDevel
          zigradMlirExtDevel
          tvmDevel.dev
          cudaCompileHeaders
          pkgs.mkl
        ]
        ++ pkgs.lib.optionals (mirageRuntime != null) [
          mirageRuntime
        ];
    };

    zigradExternalSdkRuntimeFullDevel = pkgs.symlinkJoin {
      name = "zigrad-external-sdk-runtime-full-devel";
      paths = [
        xlaPjrtPluginsCudaDevel
        tvmDevel
      ];
    };

    zigradExternalSdkRuntimeNoTvmDevel = pkgs.symlinkJoin {
      name = "zigrad-external-sdk-runtime-no-tvm-devel";
      paths = [
        xlaPjrtPluginsCudaDevel
      ];
    };

    # Dev: ccache + devel (save more build artifacts + NVIDIA headers)
    zigradExternalSdkDevel = pkgs.symlinkJoin {
      name = "zigrad-external-sdk-devel";
      paths = [
        zigradExternalSdkBuildDevel
        zigradExternalSdkRuntimeFullDevel
      ];
    };

    zigrad = pkgs.callPackage ../zigrad.nix {
      inherit zigradSrc;
      version = zigradVersion;
      sdk = zigradExternalSdkBuild;
      optimize = "ReleaseFast";
    };

    zigradDevel = pkgs.callPackage ../zigrad.nix {
      inherit zigradSrc;
      version = zigradVersion;
      sdk = zigradExternalSdkBuildDevel;
      optimize = "ReleaseSafe";
    };

    hostCheckTvmRuntimeFullCompiler = pkgs.writeShellScriptBin "zigrad-check-tvm-runtime-full-compiler" ''
      set -euo pipefail

      runtime_root="${zigradExternalSdkRuntimeFullDevel}"
      export LD_LIBRARY_PATH="$runtime_root/lib:$runtime_root/runtime/sys/lib:$runtime_root/runtime/nvidia/nvrtc/lib:$runtime_root/runtime/nvidia/nvjitlink/lib:/run/opengl-driver/lib:''${LD_LIBRARY_PATH:-}"

      exec ${zigrad}/bin/zigrad tvm-check-compiler-load "$@"
    '';

    mlirExtSrc = let
      fs = pkgs.lib.fileset;
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
      devel = false;
    };

    zigradMlirExtDevel = pkgs.callPackage ../zigrad-mlir-ext.nix {
      inherit xlaMlirStablehloCapiSdk llvm;
      stdenv = pkgs.ccacheStdenv;
      src = mlirExtSrc;
      devel = true;
    };

    # ------------------------------------------------------------------
    # Production PJRT C API plugins from XLA.
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
      copyNcclNvshmem = true;
      copyCudaTools = true;
      copyLibdevice = true;
      cpuMathLibrary = "onednn-thunk";
      cpuNativeTuning = true;
      depsHash = xlaPjrtDepsHashes.cuda-onednn-thunk-native;
    };

    # Devel aliases currently use the same Bazel artifacts.
    xlaPjrtPluginsDevel = xlaPjrtPlugins;
    xlaPjrtPluginsCudaDevel = xlaPjrtPluginsCuda;

    # TVM with LLVM 22 (built from XLA-pinned sources).
    # Uses shared LLVM to match SDK, avoiding pass registry conflicts.
    tvm = pkgs.callPackage ../tvm.nix {
      inherit
        cudaPackages
        gccHost
        llvm
        ;
      inherit (cudaCfg) cudaArchitectures;
      cudaSupport = true;
      devel = false;
    };

    tvmDevel = pkgs.callPackage ../tvm.nix {
      inherit
        cudaPackages
        gccHost
        llvm
        ;
      inherit (cudaCfg) cudaArchitectures;
      cudaSupport = true;
      devel = true;
    };

    tvmCpu = pkgs.callPackage ../tvm.nix {
      inherit
        cudaPackages
        gccHost
        llvm
        ;
      inherit (cudaCfg) cudaArchitectures;
      cudaSupport = false;
    };
  in {
    # secondary deliverable are hermetic packages + explicit run wrappers (secondary bc we dont rly have a finished thing rn)
    packages =
      {
        zigrad = zigrad;
        zigrad-devel = zigradDevel;
        zigrad-check-tvm-runtime-full-compiler = hostCheckTvmRuntimeFullCompiler;
      }
      // (pkgs.lib.optionalAttrs (mirageRuntime != null) {
        mirage-runtime = mirageRuntime;
      })
      // {
        # Convenience aggregate and primary target.
        #   others are individually targetable mostly for development reasons
        zigrad-external-sdk = zigradExternalSdk;
        zigrad-external-sdk-build = zigradExternalSdkBuild;
        zigrad-external-sdk-runtime-full = zigradExternalSdkRuntimeFull;
        zigrad-external-sdk-runtime-no-tvm = zigradExternalSdkRuntimeNoTvm;

        # Dev target: ccache + devel
        zigrad-external-sdk-devel = zigradExternalSdkDevel;
        zigrad-external-sdk-build-devel = zigradExternalSdkBuildDevel;
        zigrad-external-sdk-runtime-full-devel = zigradExternalSdkRuntimeFullDevel;
        zigrad-external-sdk-runtime-no-tvm-devel = zigradExternalSdkRuntimeNoTvmDevel;

        # PJRT C API plugins (buildBazelPackage)
        xla-pjrt-plugins = xlaPjrtPlugins;
        xla-pjrt-plugins-devel = xlaPjrtPluginsDevel;
        xla-pjrt-plugins-cuda = xlaPjrtPluginsCuda;
        xla-pjrt-plugins-cuda-devel = xlaPjrtPluginsCudaDevel;

        # Compile-time SDK (PJRT headers + MLIR + StableHLO)
        xla-mlir-stablehlo-capi-sdk = xlaMlirStablehloCapiSdk;

        # Compile-time SDK - Dev target: ccache + devel
        xla-mlir-stablehlo-capi-sdk-devel = xlaMlirStablehloCapiDevel;

        tvm = tvm;
        tvm-dev = tvm.dev;
        tvm-cpu = tvmCpu;
        tvm-devel = tvmDevel;

        # LLVM 22 built from XLA-pinned sources
        llvm = llvm;

        gen-clangd = targets.editor.clangd;
        gen-nvim = targets.editor.nvim;
        # TODO: hermetic zig build/run targets
        # m1 = targets.m1.build;
        m4 = targets.zigrad-m4.build;
      };

    apps =
      (pkgs.lib.optionalAttrs (targets ? example-cuda) {
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
