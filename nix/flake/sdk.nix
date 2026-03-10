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
    mirageRuntime =
      if builtins.hasAttr system inputs.mpk.packages
      then inputs.mpk.packages.${system}.mirage-runtime
      else null;

    zigradSrc = import ../source-filter.nix {
      inherit (pkgs) lib;
      root = ../..;
    };

    inherit
      (import ../targets.nix {
        inherit pkgs cudaPackages gccHost;
        inherit (cudaCfg) cudaArchitectures;
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

    cudaCompileHeaders = pkgs.runCommand "cuda-compile-headers" {} ''
      mkdir -p "$out/include"
      cp -as ${cudaPackages.cudatoolkit}/include/. "$out/include/"
    '';

    # LLVM 22 from XLA-pinned sources. Shared by MLIR SDK and TVM.
    llvm = pkgs.callPackage ../llvm.nix {inherit xlaSrc llvmSrc;};

    xlaMlirStablehloCapiSdk = pkgs.callPackage ../xla-mlir-stablehlo-capi-sdk.nix {
      inherit xlaSrc stablehloSrc llvm;
    };

    zigradMlirExt = pkgs.callPackage ../zigrad-mlir-ext.nix {
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
            ../../shim/test
          ];
        };
    };

    xlaPjrtPlugins = pkgs.callPackage ../xla-pjrt-runtime-bazel.nix {
      inherit xlaSrc;
      cudaSupport = false;
      cpuMathLibrary = "onednn";
      cpuNativeTuning = true;
      depsHash = "sha256-vpI+i27sWrNS/qeICNav8lZJHcGsnx+C+e58oAyd3oE=";
    };

    xlaPjrtPluginsCuda = pkgs.callPackage ../xla-pjrt-runtime-bazel.nix {
      inherit xlaSrc;
      inherit (cudaCfg) cudaArchitectures cudaVersion;
      cudaSupport = true;
      cpuMathLibrary = "onednn-thunk";
      cpuNativeTuning = true;
      depsHash = "sha256-ivbrLtStbE1IW9hTiqyL0KoBdK9IZRPXcDmokH01eCE=";
    };

    cudaRedist = pkgs.callPackage ../cuda-redist.nix {inherit (cudaCfg) cudaVersion;};

    # TVM with shared LLVM 22 (avoids pass registry conflicts with MLIR SDK).
    tvm = pkgs.callPackage ../tvm.nix {
      inherit cudaPackages gccHost llvm;
      inherit (cudaCfg) cudaArchitectures;
      cudaSupport = true;
    };

    tvmCpu = pkgs.callPackage ../tvm.nix {
      inherit cudaPackages gccHost llvm;
      inherit (cudaCfg) cudaArchitectures;
      cudaSupport = false;
    };

    # IREE: BYO-LLVM from iree-org fork (diverges from XLA-pinned llvm).
    ireeLlvm = pkgs.callPackage ../iree-llvm.nix {inherit ireeLlvmSrc;};
    ireeCompiler = pkgs.callPackage ../iree-compiler.nix {
      inherit ireeSrc ireeStablehloSrc ireeFlatccSrc ireeBenchmarkSrc ireeLlvm;
    };
    ireeRuntime = pkgs.callPackage ../iree-runtime.nix {
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
          [pjrtHeaders]
          ++ lib.optional f.mlir xlaMlirStablehloCapiSdk
          ++ lib.optional f.mlir zigradMlirExt
          ++ lib.optional f.tvm tvm.dev
          ++ lib.optional f.gpu cudaCompileHeaders
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
    };

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
        xla-mlir-stablehlo-capi-sdk = xlaMlirStablehloCapiSdk;
        xla-pjrt-plugins = xlaPjrtPlugins;
        xla-pjrt-plugins-cuda = xlaPjrtPluginsCuda;
        iree-llvm = ireeLlvm;
        iree-compiler = ireeCompiler;
        iree-runtime = ireeRuntime;
        pjrt-headers = pjrtHeaders;
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
