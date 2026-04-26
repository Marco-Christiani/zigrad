# nix/flake/sdk.nix
#
# SDK assembly: leaf derivations, mkSdk compositor, named presets,
# package exports, and apps.
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
    # Pick mpk's CUDA-pinned variant matching cudaCfg. Without this, the default
    #  mpk attr resolves to nixpkgs's cudaPackages_12 alias (currently 12.8),
    #  which leaks ~7 GiB of CUDA 12.8 into the closure alongside our 12.9.
    mirageVariant = "cuda${builtins.replaceStrings ["."] ["-"] (pkgs.lib.versions.majorMinor cudaCfg.cudaVersion)}-mirage-runtime";
    mirageRuntime =
      if builtins.hasAttr system inputs.mpk.packages
        && builtins.hasAttr mirageVariant inputs.mpk.packages.${system}
      then inputs.mpk.packages.${system}.${mirageVariant}
      else null;

    zigradSrc = import ../helpers/source-filter.nix {
      inherit (pkgs) lib;
      root = ../..;
    };

    inherit
      (import ../helpers/targets.nix {
        inherit pkgs cudaToolkit gccHost;
        inherit (pkgs) zig;
        src = zigradSrc;
        zigradExternalSdk = sdkProfiles.full-gpu.full;
      })
      targets
      ;

    # Leaf derivations

    # PJRT + XLA FFI headers (pure source copy, zero build cost).
    pjrtHeaders = pkgs.runCommand "pjrt-xla-ffi-headers" {} ''
      mkdir -p $out/include/xla/pjrt/c $out/include/xla/ffi/api
      cp ${xlaSrc}/xla/pjrt/c/*.h $out/include/xla/pjrt/c/
      cp ${xlaSrc}/xla/ffi/api/*.h $out/include/xla/ffi/api/
    '';

    # XLA proto schemas (for protobuf decode tooling, zero build cost).
    xlaProtos = pkgs.runCommand "xla-protos" {} ''
      mkdir -p $out/proto/xla/service
      cp ${xlaSrc}/xla/xla.proto $out/proto/xla/
      cp ${xlaSrc}/xla/xla_data.proto $out/proto/xla/
      cp ${xlaSrc}/xla/service/hlo.proto $out/proto/xla/service/
      cp ${xlaSrc}/xla/service/metrics.proto $out/proto/xla/service/
    '';

    # LLVM 22 from XLA-pinned sources. Shared by MLIR SDK and TVM.
    llvm = pkgs.callPackage ../packages/llvm.nix {inherit xlaSrc llvmSrc;};

    xlaMlirStablehloCapiSdk = pkgs.callPackage ../packages/xla-mlir-stablehlo-capi-sdk.nix {
      inherit xlaSrc stablehloSrc llvm;
    };

    zigradMlirExt = pkgs.callPackage ../packages/zigrad-mlir-ext.nix {
      inherit xlaMlirStablehloCapiSdk llvm;
      src = let
        fs = lib.fileset;
      in
        fs.toSource {
          root = ../../shim;
          fileset = fs.unions [
            ../../shim/CMakeLists.txt
            ../../shim/mlir_ext.cc
            ../../shim/zigrad
            ../../shim/dev
            ../../shim/test
          ];
        };
    };

    xlaPjrtPlugins = pkgs.callPackage ../packages/xla-pjrt-runtime-bazel.nix {
      inherit xlaSrc;
      cudaSupport = false;
      cpuMathLibrary = "onednn";
      cpuNativeTuning = true;
      depsHash = "sha256-vpI+i27sWrNS/qeICNav8lZJHcGsnx+C+e58oAyd3oE=";
    };

    xlaPjrtPluginsCuda = pkgs.callPackage ../packages/xla-pjrt-runtime-bazel.nix {
      inherit xlaSrc;
      inherit (cudaCfg) cudaVersion;
      cudaSupport = true;
      cpuMathLibrary = "onednn-thunk";
      cpuNativeTuning = true;
      depsHash = "sha256-nOcNUVt5HFO8eC2C9Ubjjwn1dnKnCuWZISZcrhbczd0=";
    };

    cudaRedist = pkgs.callPackage ../packages/cuda-redist.nix {inherit (cudaCfg) cudaVersion;};
    # Cudatoolkit-style layout (bin/nvcc, include/, lib/, lib/stubs/) for
    #  build-time consumers. Production-path: never uses pkgs.cudaPackages.
    cudaToolkit = cudaRedist.dev;

    # TVM with shared LLVM 22 (avoids pass registry conflicts with MLIR SDK).
    # cudaArchitectures left at upstream default (no TVM_CUDA_ARCH override);
    #  TVM compiles kernels via NVRTC at runtime against the actual GPU, so the
    #  build-time arch hint matters only for AOT paths we don't use.
    tvm = pkgs.callPackage ../packages/tvm.nix {
      inherit cudaToolkit gccHost llvm;
      # cuda-redist.out has a flat lib/ symlink farm pointing into the runtime
      #  layout; passing it as cudaRuntime makes libtvm.so's rpath reference
      #  the runtime layout instead of cudaToolkit (= cuda-redist.dev). Keeps
      #  dev (with its build-time .a files and unused link-time .so) out of
      #  TVM's runtime closure.
      cudaRuntime = cudaRedist.out;
      cudaSupport = true;
    };

    tvmCpu = pkgs.callPackage ../packages/tvm.nix {
      inherit gccHost llvm;
      cudaSupport = false;
    };

    # IREE: BYO-LLVM from iree-org fork (diverges from XLA-pinned llvm).
    ireeLlvm = pkgs.callPackage ../packages/iree/llvm.nix {inherit ireeLlvmSrc;};
    ireeCompiler = pkgs.callPackage ../packages/iree/compiler.nix {
      inherit ireeSrc ireeStablehloSrc ireeFlatccSrc ireeBenchmarkSrc ireeLlvm;
      withCli = true;
    };
    ireeRuntime = pkgs.callPackage ../packages/iree/runtime.nix {
      inherit ireeSrc ireeStablehloSrc ireeFlatccSrc ireeBenchmarkSrc ireeLlvm;
    };

    # SDK compositor: feature flags -> { compile, runtime, full }.
    # Takes a plain attrset merged with defaults (avoids shadowing `tvm`/`iree` derivations).
    mkSdk = features: let
      f =
        {
          mlir = true;
          iree = true;
          tvm = true;
          gpu = true;
          mirage = true;
          mkl = false;
        }
        // features;
      hasMirage = f.mirage && mirageRuntime != null;
    in rec {
      compile = pkgs.symlinkJoin {
        name = "zigrad-sdk-compile";
        paths =
          [pjrtHeaders xlaProtos]
          # .dev contains headers + full lib closure; needed for build-time
          #  consumers (zig build, downstream cmake). .out has only the minimal
          #  DT_NEEDED runtime closure (no headers).
          ++ lib.optional f.mlir xlaMlirStablehloCapiSdk.dev
          ++ lib.optional f.mlir zigradMlirExt
          ++ lib.optional f.tvm tvm.dev
          # cuda-redist's dev output supplies headers + nvcc + lib + stubs in a
          #  cudatoolkit-style layout. Replaces the prior cudaCompileHeaders
          #  thin wrapper around pkgs.cudaPackages.cudatoolkit/include.
          ++ lib.optional f.gpu cudaToolkit
          ++ lib.optional f.iree ireeCompiler
          ++ lib.optional f.iree ireeRuntime
          ++ lib.optional hasMirage mirageRuntime
          ++ lib.optional f.mkl pkgs.mkl;
      };

      runtime = pkgs.symlinkJoin {
        name = "zigrad-sdk-runtime";
        paths =
          [
            (
              if f.gpu
              then xlaPjrtPluginsCuda
              else xlaPjrtPlugins
            )
          ]
          ++ lib.optional f.gpu cudaRedist
          ++ lib.optional f.tvm tvm
          ++ lib.optional hasMirage mirageRuntime;
      };

      full = pkgs.symlinkJoin {
        name = "zigrad-sdk";
        paths = [compile runtime];
      };
    };

    sdkProfiles = {
      full-gpu = mkSdk {};
      mlir-cpu = mkSdk {
        tvm = false;
        gpu = false;
      };
      iree-cpu = mkSdk {
        mlir = false;
        iree = true;
        tvm = false;
        gpu = false;
      };
      tvm-gpu = mkSdk {mlir = false;};
      minimal = mkSdk {
        mlir = false;
        tvm = false;
        gpu = false;
      };
      full-benchmark = mkSdk {
        mlir = true;
        iree = true;
        tvm = true;
        gpu = true;
        mirage = true;
        mkl = true;
      };
    };

    zigrad = pkgs.callPackage ../packages/zigrad.nix {
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
        inherit zigrad;
        zigrad-check-tvm-runtime-full-compiler = hostCheckTvmRuntimeFullCompiler;
      }
      // (lib.optionalAttrs (mirageRuntime != null) {
        mirage-runtime = mirageRuntime;
      })
      // lib.concatMapAttrs (name: profile: {
        "zigrad-sdk-${name}-compile" = profile.compile;
        "zigrad-sdk-${name}-runtime" = profile.runtime;
        "zigrad-sdk-${name}" = profile.full;
      })
      sdkProfiles
      // {
        inherit llvm tvm;
        tvm-dev = tvm.dev;
        tvm-cpu = tvmCpu;
        cuda-redist = cudaRedist;
        cuda-redist-dev = cudaRedist.dev;
        xla-mlir-stablehlo-capi-sdk = xlaMlirStablehloCapiSdk;
        xla-mlir-stablehlo-capi-sdk-dev = xlaMlirStablehloCapiSdk.dev;
        xla-pjrt-plugins = xlaPjrtPlugins;
        xla-pjrt-plugins-cuda = xlaPjrtPluginsCuda;
        iree-llvm = ireeLlvm;
        iree-compiler = ireeCompiler;
        iree-runtime = ireeRuntime;
        pjrt-headers = pjrtHeaders;
        # MLIR extension: production .so under .out (consumed by SDK profiles
        #  via mkSdk's f.mlir branch), LSP server binary under .dev (consumed
        #  by the devshell). Single source build, two consumption surfaces.
        zigrad-mlir-ext = zigradMlirExt;
        zigrad-mlir-ext-dev = zigradMlirExt.dev;
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
