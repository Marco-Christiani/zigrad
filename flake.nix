{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

    mpk = {
      url = "path:/home/marco/flakes/mpk";
      inputs.nixpkgs.follows = "nixpkgs";
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
  };
  outputs = {
    self,
    nixpkgs,
    mpk,
    pyproject-nix,
    uv2nix,
    pyproject-build-systems,
  }: let
    systems = [
      "x86_64-linux"
      "aarch64-linux"
    ];

    forAllSystems = f:
      builtins.listToAttrs (
        map (system: {
          name = system;
          value = f system;
        })
        systems
      );

    cudaCfg = import ./nix/cuda.nix;

    colors = {
      yellow = "\\033[33m";
      reset = "\\033[0m";
      green = "\\033[32m";
    };

    mkFor = system: let
      pkgs = import nixpkgs {
        inherit system;
        overlays = [
          (import ./nix/overlays/ccache.nix)
        ];
        config = {
          allowUnfree = true;
          # note to self: avoid enabling cudaSupport globally unless you need nixpkgs packages to flip CUDA paths.
          # it might have wide-reaching effects on unrelated packages.
          # cudaSupport = true;
        };
      };

      cudaPackages = pkgs.${cudaCfg.cudaPackagesAttr};
      gccHost = pkgs.${cudaCfg.gccHostAttr};
      mirageRuntime = if builtins.hasAttr system mpk.packages then mpk.packages.${system}.mirage-runtime else null;

      zigradSrc = import ./nix/source-filter.nix {
        lib = pkgs.lib;
        root = ./.;
      };

      src = zigradSrc;

      inherit
        (import ./nix/targets.nix {
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

      lockFile = ./nix/lock.json;

      # buildBazelPackage fetchAttrs hashes for xla-pjrt-runtime-bazel.nix.
      # These hashes are configuration-specific (CUDA on/off, cpuMathLibrary,
      # native tuning flags, etc) and must be maintained per combination.
      xlaPjrtDepsHashes = {
        cpu-onednn-native = "sha256-vpI+i27sWrNS/qeICNav8lZJHcGsnx+C+e58oAyd3oE=";
        cuda-onednn-thunk-native = "sha256-EFE6NyvyOoereFfwwExFu03B6IjfjrBWD5ri6S3F/9Y=";
      };

      # Compile-time CUDA headers used by Zig @cImport("nvrtc.h").
      # Keep this isolated from runtime CUDA DSOs, which come from PJRT runtime bundles.
      cudaCompileHeaders = pkgs.runCommand "cuda-compile-headers" {} ''
        mkdir -p "$out/include"
        cp -as ${cudaPackages.cudatoolkit}/include/. "$out/include/"
      '';

      xlaMlirStablehloCapiSdk = pkgs.callPackage ./nix/xla-mlir-stablehlo-capi-sdk.nix {
        inherit lockFile;
      };

      xlaMlirStablehloCapiDevel = pkgs.callPackage ./nix/xla-mlir-stablehlo-capi-sdk.nix {
        inherit lockFile;
        stdenv = pkgs.ccacheStdenv;
        devel = true;
      };

      # LLVM 22 built from XLA-pinned sources. Shared by SDK and TVM to ensure
      # they use the same LLVM version (same pass registry, no ABI conflicts).
      llvm = pkgs.callPackage ./nix/llvm.nix {
        inherit lockFile;
      };

      # SDK profiles:
      # - build: compile-time headers/libs (TVM + CUDA headers, no TVM runtime DSOs)
      # - runtime-full: includes TVM runtime/compiler DSOs
      # - runtime-no-tvm: intentionally excludes TVM runtime/compiler DSOs
      # - aggregate: build + runtime-full for default ergonomic workflows
      zigradExternalSdkBuild = pkgs.symlinkJoin {
        name = "zigrad-external-sdk-build";
        paths = [
          xlaMlirStablehloCapiSdk
          tvm.dev
          cudaCompileHeaders
          pkgs.mkl
        ] ++ pkgs.lib.optionals (mirageRuntime != null) [
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
          # zigradMlirShim
        ];
      };
      sdkRoot = toString zigradExternalSdk;

      zigradExternalSdkBuildDevel = pkgs.symlinkJoin {
        name = "zigrad-external-sdk-build-devel";
        paths = [
          xlaMlirStablehloCapiDevel
          tvmDevel.dev
          cudaCompileHeaders
          pkgs.mkl
        ] ++ pkgs.lib.optionals (mirageRuntime != null) [
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
          # zigradMlirShimDevel
        ];
      };
      sdkRootDevel = toString zigradExternalSdkDevel;

      zigrad = pkgs.callPackage ./nix/zigrad.nix {
        zigradSrc = zigradSrc;
        sdk = zigradExternalSdkBuild;
        optimize = "ReleaseFast";
      };

      zigradDevel = pkgs.callPackage ./nix/zigrad.nix {
        zigradSrc = zigradSrc;
        sdk = zigradExternalSdkBuildDevel;
        optimize = "ReleaseSafe";
      };

      zigradTests = pkgs.callPackage ./nix/zigrad.nix {
        zigradSrc = zigradSrc;
        sdk = zigradExternalSdkBuildDevel;
        optimize = "ReleaseSafe";
        runTests = true;
      };

      checkTvmRuntimeFullFfi = pkgs.runCommand "check-zigrad-tvm-runtime-full-ffi" {
        nativeBuildInputs = [
          zigrad
        ];
      } ''
        set -euo pipefail
        export HOME="$TMPDIR"
        runtime_root="${zigradExternalSdkRuntimeFullDevel}"
        export LD_LIBRARY_PATH="$runtime_root/lib:$runtime_root/runtime/sys/lib:$runtime_root/runtime/nvidia/nvrtc/lib:$runtime_root/runtime/nvidia/nvjitlink/lib"

        ${zigrad}/bin/zigrad tvm-dump-symbols > "$TMPDIR/tvm-symbols.txt"
        test -s "$TMPDIR/tvm-symbols.txt"

        mkdir -p "$out"
        cp "$TMPDIR/tvm-symbols.txt" "$out/tvm-symbols.txt"
      '';

      hostCheckTvmRuntimeFullCompiler = pkgs.writeShellScriptBin "zigrad-check-tvm-runtime-full-compiler" ''
        set -euo pipefail

        runtime_root="${zigradExternalSdkRuntimeFullDevel}"
        export LD_LIBRARY_PATH="$runtime_root/lib:$runtime_root/runtime/sys/lib:$runtime_root/runtime/nvidia/nvrtc/lib:$runtime_root/runtime/nvidia/nvjitlink/lib:/run/opengl-driver/lib:''${LD_LIBRARY_PATH:-}"

        exec ${zigrad}/bin/zigrad tvm-check-compiler-load "$@"
      '';

      checkTvmRuntimeNoTvm = pkgs.runCommand "check-zigrad-tvm-runtime-no-tvm" {
        nativeBuildInputs = [
          zigrad
        ];
      } ''
        set -euo pipefail
        export HOME="$TMPDIR"
        runtime_root="${zigradExternalSdkRuntimeNoTvmDevel}"
        export LD_LIBRARY_PATH="$runtime_root/lib:$runtime_root/runtime/sys/lib:$runtime_root/runtime/nvidia/nvrtc/lib:$runtime_root/runtime/nvidia/nvjitlink/lib"

        if ${zigrad}/bin/zigrad tvm-dump-symbols > "$TMPDIR/stdout.txt" 2> "$TMPDIR/stderr.txt"; then
          echo "expected tvm-dump-symbols to fail without TVM runtime libraries" >&2
          exit 1
        fi

        if ! grep -Eq "(TvmLoadFailed|failed to load TVM FFI runtime|dlopen)" "$TMPDIR/stderr.txt"; then
          echo "expected loader diagnostics in stderr" >&2
          cat "$TMPDIR/stderr.txt" >&2
          exit 1
        fi

        mkdir -p "$out"
        cp "$TMPDIR/stderr.txt" "$out/tvm-missing-stderr.txt"
      '';

      # zigradMlirShim = pkgs.callPackage ./nix/zigrad-mlir-shim.nix {
      #   inherit xlaMlirStablehloCapiSdk;
      #   src = shimSrc;
      #   devel = false;
      # };
      #
      # zigradMlirShimDevel = pkgs.callPackage ./nix/zigrad-mlir-shim.nix {
      #   inherit xlaMlirStablehloCapiSdk;
      #   stdenv = pkgs.ccacheStdenv;
      #   src = shimSrc;
      #   devel = true;
      # };

      # ------------------------------------------------------------------
      # Production PJRT C API plugins from XLA.
      xlaPjrtPlugins = pkgs.callPackage ./nix/xla-pjrt-runtime-bazel.nix {
        inherit lockFile;
        cudaSupport = false;
        cpuMathLibrary = "onednn";
        cpuNativeTuning = true;
        depsHash = xlaPjrtDepsHashes.cpu-onednn-native;
      };

      xlaPjrtPluginsCuda = pkgs.callPackage ./nix/xla-pjrt-runtime-bazel.nix {
        inherit lockFile;
        inherit (cudaCfg) cudaArchitectures cudaVersion;
        cudaSupport = true;
        copyNcclNvshmem = true;
        copyCudaTools = true;
        copyLibdevice = true;
        # cpuMathLibrary = "onednn";
        cpuMathLibrary = "onednn-thunk";
        cpuNativeTuning = true;
        depsHash = xlaPjrtDepsHashes.cuda-onednn-thunk-native;
      };

      # Devel aliases currently use the same Bazel artifacts.
      xlaPjrtPluginsDevel = xlaPjrtPlugins;
      xlaPjrtPluginsCudaDevel = xlaPjrtPluginsCuda;

      # Legacy non-buildBazelPackage path.
      xlaPjrtPluginsLegacy = pkgs.callPackage ./nix/xla-pjrt-runtime.nix {
        inherit lockFile;
        devel = false;
        cudaSupport = false;
        cudaPackages = null;
        persistentBazelOutputBase = false;
        cpuMathLibrary = "onednn-thunk";
        cpuNativeTuning = true;
      };

      xlaPjrtPluginsCudaLegacy = pkgs.callPackage ./nix/xla-pjrt-runtime.nix {
        inherit lockFile;
        inherit (cudaCfg) cudaArchitectures cudaVersion;
        devel = false;
        cudaSupport = true;
        copyNcclNvshmem = true;
        copyCudaTools = true;
        copyLibdevice = true;
        cudaPackages = null;
        useCudaStdenv = false;
        persistentBazelOutputBase = false;
        cpuMathLibrary = "onednn-thunk";
        cpuNativeTuning = true;
      };

      xlaPjrtPluginsLegacyDevel = pkgs.callPackage ./nix/xla-pjrt-runtime.nix {
        inherit lockFile;
        stdenv = pkgs.ccacheStdenv;
        devel = true;
        cudaSupport = false;
        cudaPackages = null;
        persistentBazelOutputBase = true;
        cpuMathLibrary = "onednn-thunk";
        cpuNativeTuning = true;
      };

      xlaPjrtPluginsCudaLegacyDevel = pkgs.callPackage ./nix/xla-pjrt-runtime.nix {
        inherit lockFile;
        inherit (cudaCfg) cudaArchitectures cudaVersion;
        stdenv = pkgs.ccacheStdenv;
        devel = true;
        cudaSupport = true;
        copyNcclNvshmem = true;
        copyCudaTools = true;
        copyLibdevice = true;
        cudaPackages = null;
        useCudaStdenv = false;
        persistentBazelOutputBase = true;
        cpuMathLibrary = "onednn-thunk";
        cpuNativeTuning = true;
      };

      # Compatibility aliases.
      xlaPjrtPluginsBazel = xlaPjrtPlugins;
      xlaPjrtPluginsBazelCuda = xlaPjrtPluginsCuda;

      # TVM with LLVM 22 (built from XLA-pinned sources).
      # Uses shared LLVM to match SDK, avoiding pass registry conflicts.
      tvm = pkgs.callPackage ./nix/tvm.nix {
        inherit
          cudaPackages
          gccHost
          llvm
          ;
        inherit (cudaCfg) cudaArchitectures;
        cudaSupport = true;
        devel = false;
      };

      tvmDevel = pkgs.callPackage ./nix/tvm.nix {
        inherit
          cudaPackages
          gccHost
          llvm
          ;
        inherit (cudaCfg) cudaArchitectures;
        cudaSupport = true;
        devel = true;
      };

      tvmCpu = pkgs.callPackage ./nix/tvm.nix {
        inherit
          cudaPackages
          gccHost
          llvm
          ;
        inherit (cudaCfg) cudaArchitectures;
        cudaSupport = false;
      };

      # ------------------------------------------------------------------
      baseDevShellPkgs = with pkgs; [
        zig
        # zls
        go-task
        binutils
        patchelf
        git
        gccHost
        clang
      ];

      pyShellPkgs = pkgs.callPackage ./nix/pydev.nix {
        inherit
          system
          pyproject-nix
          uv2nix
          pyproject-build-systems
          cudaPackages
          ;
        py = pkgs.python312;
        py-pkgs = pkgs.python312Packages;
      };

      pythonJaxCudaOverride = pkgs.python312.override {
        packageOverrides = self: super: {
          jax = super.jax.override {
            # inherit cudaPackages;
            cudaSupport = true;
          };
        };
      };
    in {
      # devshells intended purpose is really just fast iteration and pinned toolchain with
      #   relaxed hermeticity requirements as needed for productivity.
      devShells = {
        default = pkgs.mkShellNoCC {
          packages = pyShellPkgs.out.packages ++ baseDevShellPkgs ++ [zigradExternalSdkDevel];
          env =
            pyShellPkgs.out.env
            // {
              ZG_EXTERNAL_SDK_ROOT = sdkRootDevel;
              # PJRT_PLUGIN_PATH = "${sdkRootDevel}/runtime/jax_plugins/xla_cuda13/xla_cuda_plugin.so";
              PJRT_CPU_PLUGIN_PATH = "${sdkRootDevel}/runtime/xla/pjrt/c/pjrt_c_api_cpu_plugin.so";
              PJRT_GPU_PLUGIN_PATH = "${sdkRootDevel}/runtime/xla/pjrt/c/pjrt_c_api_gpu_plugin.so";
              PYTHONPATH = "${sdkRootDevel}/python";
              CUDA_HOME = "${cudaPackages.cudatoolkit}";
              # NVRTC include paths for nix-compatible CUDA compilation
              NIX_GLIBC_INCLUDE = "${pkgs.stdenv.cc.libc.dev}/include";
              NIX_GCC_INCLUDE = "${pkgs.stdenv.cc.cc}/lib/gcc/${pkgs.stdenv.hostPlatform.config}/${pkgs.lib.getVersion pkgs.stdenv.cc.cc}/include";
            };
          shellHook = ''
            export REPO_ROOT=$(git rev-parse --show-toplevel)

            [[ -f "$PJRT_CPU_PLUGIN_PATH" ]]
            cpu_plugin_exists=$?

            [[ -f "$PJRT_GPU_PLUGIN_PATH" ]]
            gpu_plugin_exists=$?

            if (( cpu_plugin_exists != 0 )); then
              # printf "%b[WARNING]%b PJRT_CPU_PLUGIN_PATH=%s does not exist. Leaving the env variable set but you may need to materialize this.\n" \
              #   "${colors.yellow}" "${colors.reset}" "$PJRT_CPU_PLUGIN_PATH"
              true
            fi

            if (( gpu_plugin_exists != 0 )); then
              # printf "%b[WARNING]%b PJRT_GPU_PLUGIN_PATH=%s does not exist. Leaving the env variable set but you may need to materialize this.\n" \
              #   "${colors.yellow}" "${colors.reset}" "$PJRT_GPU_PLUGIN_PATH"
              true
            fi

            # Selection logic:
            # - If both plugins exist, GPU is preferred
            # - If only GPU exists, use GPU
            # - Otherwise fall back to CPU (warnings already emitted)
            if (( gpu_plugin_exists == 0 )); then
              if (( cpu_plugin_exists == 0 )); then
                # printf "%b[INFO]%b Both CPU and GPU plugins exist. Selecting GPU plugin as the preferred option.\n" \
                #   "${colors.yellow}" "${colors.reset}"
                true
              fi
              PJRT_PLUGIN_PATH="$PJRT_GPU_PLUGIN_PATH"
            else
              PJRT_PLUGIN_PATH="$PJRT_CPU_PLUGIN_PATH"
            fi
            export PJRT_PLUGIN_PATH
            # zig is not happy about -fmacro-prefix-map
            unset NIX_CFLAGS_COMPILE
          '';
        };

        # A purer target that uses the real derivations with devel=true, no extra copies, no ccache.
        #   Can sandbox.
        pure = pkgs.mkShellNoCC {
          packages = baseDevShellPkgs ++ [zigradExternalSdk];
          ZG_EXTERNAL_SDK_ROOT = sdkRoot;
          PJRT_PLUGIN_PATH = "${sdkRoot}/runtime/xla/pjrt/c/pjrt_c_api_gpu_plugin.so";

          shellHook = ''
            if [[ ! -f $PJRT_PLUGIN_PATH ]]; then
              printf "${colors.yellow}[WARNING]${colors.reset} PJRT_PLUGIN_PATH=$PJRT_PLUGIN_PATH does not exist. \
                      Leaving the env variable set but you may need to materialize this.\n"
            fi
            printf "SDK path: ZG_EXTERNAL_SDK_ROOT=$ZG_EXTERNAL_SDK_ROOT"
          '';
        };

        profiling = pkgs.mkShellNoCC {
          packages =
            pyShellPkgs.out.packages
            ++ baseDevShellPkgs
            ++ [
              zigradExternalSdkDevel
              cudaPackages.nsight_systems # nix-du: ~1.1 / manual diffing: ~2.3GiB / nix-tree: NAR Size: 8.11 KiB | Closure Size: 15.39 MiB | Added Size: 93.75 KiB
              cudaPackages.nsight_compute # nix-du: ~1.3 / manual diffing: 2.5GiB / NAR Size: 4.93 KiB | Closure Size: 15.24 MiB | Added Size: 16.35 KiB
            ];
          env =
            pyShellPkgs.out.env
            // {
              ZG_EXTERNAL_SDK_ROOT = sdkRootDevel;
              PJRT_CPU_PLUGIN_PATH = "${sdkRootDevel}/runtime/xla/pjrt/c/pjrt_c_api_cpu_plugin.so";
              PJRT_GPU_PLUGIN_PATH = "${sdkRootDevel}/runtime/xla/pjrt/c/pjrt_c_api_gpu_plugin.so";
            };
          shellHook = ''
            export REPO_ROOT=$(git rev-parse --show-toplevel)

            [[ -f "$PJRT_CPU_PLUGIN_PATH" ]]
            cpu_plugin_exists=$?

            [[ -f "$PJRT_GPU_PLUGIN_PATH" ]]
            gpu_plugin_exists=$?

            if (( gpu_plugin_exists == 0 )); then
              PJRT_PLUGIN_PATH="$PJRT_GPU_PLUGIN_PATH"
            else
              PJRT_PLUGIN_PATH="$PJRT_CPU_PLUGIN_PATH"
            fi
            export PJRT_PLUGIN_PATH
          '';
        };
      };

      # secondary deliverable are hermetic packages + explicit run wrappers (secondary bc we dont rly have a finished thing rn)
      packages = {
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

        # ----------------------------------------------------------------
        # Bazel PJRT plugin build
        xla-pjrt-plugins = xlaPjrtPlugins;

        # Bazel PJRT plugin build - Dev target: ccache + devel
        xla-pjrt-plugins-devel = xlaPjrtPluginsDevel;

        # Bazel PJRT plugin build - CUDA
        xla-pjrt-plugins-cuda = xlaPjrtPluginsCuda;

        # Bazel PJRT plugin build - CUDA + devel
        xla-pjrt-plugins-cuda-devel = xlaPjrtPluginsCudaDevel;
        # ----------------------------------------------------------------

        # Legacy PJRT plugin build path.
        xla-pjrt-plugins-legacy = xlaPjrtPluginsLegacy;
        xla-pjrt-plugins-legacy-cuda = xlaPjrtPluginsCudaLegacy;
        xla-pjrt-plugins-legacy-devel = xlaPjrtPluginsLegacyDevel;
        xla-pjrt-plugins-legacy-cuda-devel = xlaPjrtPluginsCudaLegacyDevel;

        # Compile-time SDK (PJRT headers + MLIR + StableHLO)
        xla-mlir-stablehlo-capi-sdk = xlaMlirStablehloCapiSdk;

        # Comptile-time SDK - Dev target: ccache + devel
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

        # new version with buildBazelPackage
        xla-pjrt-plugins-bazel = xlaPjrtPluginsBazel;
        xla-pjrt-plugins-bazel-cuda = xlaPjrtPluginsBazelCuda;
      };

      checks = {
        zigrad-build = zigrad;
        zigrad-unit-tests = zigradTests;
        tvm-runtime-full-ffi = checkTvmRuntimeFullFfi;
        tvm-runtime-no-tvm = checkTvmRuntimeNoTvm;
      };

      apps =
        (pkgs.lib.optionalAttrs (targets ? example-cuda) {
          example-cuda-target = {
            type = "app";
            program = "${targets.example-cuda.run}/bin/example-cuda";
          };
        })
        // {
          # m1 = {
          #   # TODO: hermetic zig build/run targets
          # };
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
  in {
    devShells = forAllSystems (s: (mkFor s).devShells);
    packages = forAllSystems (s: (mkFor s).packages);
    checks = forAllSystems (s: (mkFor s).checks);
    apps = forAllSystems (s: (mkFor s).apps);
    formatter = forAllSystems (
      system: let
        pkgs = import nixpkgs {inherit system;};
      in
        pkgs.alejandra
    );
  };
}
