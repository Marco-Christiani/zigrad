# nix/flake/devshells.nix
#
# Development shells for fast iteration with relaxed hermeticity.
{inputs, ...}: let
  colors = {
    yellow = "\\033[33m";
    reset = "\\033[0m";
    green = "\\033[32m";
  };
in {
  perSystem = {
    pkgs,
    config,
    system,
    cudaCfg,
    ...
  }: let
    inherit (inputs) uv2nix;

    cudaPackages = pkgs.${cudaCfg.cudaPackagesAttr};
    gccHost = pkgs.${cudaCfg.gccHostAttr};

    sdkRoot = toString config.packages.zigrad-external-sdk;
    sdkRootDevel = toString config.packages.zigrad-external-sdk-devel;

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

    pyShellPkgs = pkgs.callPackage ../pydev.nix {
      inherit system cudaPackages uv2nix;
      "pyproject-nix" = inputs."pyproject-nix";
      "pyproject-build-systems" = inputs."pyproject-build-systems";
      py = pkgs.python312;
      py-pkgs = pkgs.python312Packages;
    };
  in {
    # devshells intended purpose is really just fast iteration and pinned toolchain with
    #   relaxed hermeticity requirements as needed for productivity.
    devShells = {
      default = pkgs.mkShellNoCC {
        packages = pyShellPkgs.out.packages ++ baseDevShellPkgs ++ [config.packages.zigrad-external-sdk-devel];
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
        packages = baseDevShellPkgs ++ [config.packages.zigrad-external-sdk];
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
            config.packages.zigrad-external-sdk-devel
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
  };
}
