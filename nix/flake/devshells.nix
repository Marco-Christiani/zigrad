# nix/flake/devshells.nix
#
# Development shells for fast iteration with relaxed hermeticity.
{inputs, ...}: {
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

    sdkFull = toString config.packages.zigrad-sdk-full-gpu;

    baseDevShellPkgs = with pkgs; [
      zig
      # zls
      go-task
      nodejs_22
      binutils
      patchelf
      git
      gccHost
      clang
      cmake
      ninja
    ];

    pyShellPkgs = pkgs.callPackage ../pydev.nix {
      inherit system cudaPackages uv2nix;
      "pyproject-nix" = inputs."pyproject-nix";
      "pyproject-build-systems" = inputs."pyproject-build-systems";
      py = pkgs.python312;
      py-pkgs = pkgs.python312Packages;
    };

    # Shared PJRT plugin selection logic: prefer GPU if available, fall back to CPU.
    pjrtSelectHook = ''
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

    sdkEnv = {
      ZG_EXTERNAL_SDK_ROOT = sdkFull;
      PJRT_CPU_PLUGIN_PATH = "${sdkFull}/runtime/xla/pjrt/c/pjrt_c_api_cpu_plugin.so";
      PJRT_GPU_PLUGIN_PATH = "${sdkFull}/runtime/xla/pjrt/c/pjrt_c_api_gpu_plugin.so";
    };
  in {
    devShells = {
      default = pkgs.mkShellNoCC {
        packages = pyShellPkgs.out.packages ++ baseDevShellPkgs ++ [config.packages.zigrad-sdk-full-gpu];
        env =
          pyShellPkgs.out.env
          // sdkEnv
          // {
            CUDA_HOME = "${cudaPackages.cudatoolkit}";
            # NVRTC include paths for nix-compatible CUDA compilation
            NIX_GLIBC_INCLUDE = "${pkgs.stdenv.cc.libc.dev}/include";
            NIX_GCC_INCLUDE = "${pkgs.stdenv.cc.cc}/lib/gcc/${pkgs.stdenv.hostPlatform.config}/${pkgs.lib.getVersion pkgs.stdenv.cc.cc}/include";
          };
        shellHook =
          pjrtSelectHook
          + ''
            # zig is not happy about -fmacro-prefix-map
            unset NIX_CFLAGS_COMPILE
          '';
      };

      pure = pkgs.mkShellNoCC {
        packages = baseDevShellPkgs ++ [config.packages.zigrad-sdk-full-gpu];
        env = sdkEnv;
        shellHook = pjrtSelectHook;
      };

      profiling = pkgs.mkShellNoCC {
        packages =
          pyShellPkgs.out.packages
          ++ baseDevShellPkgs
          ++ [
            config.packages.zigrad-sdk-full-gpu
            cudaPackages.nsight_systems
            cudaPackages.nsight_compute
          ];
        env =
          pyShellPkgs.out.env
          // sdkEnv;
        shellHook = pjrtSelectHook;
      };
    };
  };
}
