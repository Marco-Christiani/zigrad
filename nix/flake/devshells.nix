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

    # Wrap clangd so it trusts Nix-store gcc/clang wrappers as drivers.
    # Without --query-driver, clangd refuses to query the compiler used in
    #  compile_commands.json and falls back to no system include paths,
    #  producing 'type_traits' file not found cascades in C++ TUs.
    clangdWrapped = pkgs.writeShellScriptBin "clangd" ''
      exec ${pkgs.clang-tools}/bin/clangd \
        --query-driver=/nix/store/*-gcc-wrapper-*/bin/g++,/nix/store/*-clang-wrapper-*/bin/clang++,/nix/store/*-clang-wrapper-*/bin/clang \
        "$@"
    '';

    # MLIR LSP server: ship the shim's dev output directly. The binary is
    #  installed as bin/mlir-lsp-server so editor configs using the canonical
    #  name resolve through PATH.
    zigradMlirExtDev = config.packages.zigrad-mlir-ext-dev;

    baseDevShellPkgs =
      (with pkgs; [
        zig
        zls
        zon2nix
        go-task
        nodejs_22
        binutils
        patchelf
        git
        gccHost
        clang
        cmake
        ninja
        clangdWrapped
      ])
      ++ [zigradMlirExtDev];

    # Ensure shadow precedence: clangd wrapper for query-driver, the shim
    #  dev output for mlir-lsp-server.
    lspShadowHook = ''
      export PATH="${clangdWrapped}/bin:${zigradMlirExtDev}/bin:$PATH"
    '';

    pyShellPkgs = pkgs.callPackage ./pydev.nix {
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

      # vim runtime dirs from the LLVM source tree. Editors can append these
      #  to runtimepath to get LLVM's official syntax/indent/ftplugin/ftdetect
      #  files for .mlir/.ll/.td (handles StableHLO-style assembly that the
      #  community tree-sitter grammar chokes on).
      ZG_MLIR_VIM_RT = "${inputs.llvmSrc}/mlir/utils/vim";
      ZG_LLVM_VIM_RT = "${inputs.llvmSrc}/llvm/utils/vim";
    };
  in {
    devShells = {
      default = pkgs.mkShellNoCC {
        packages = pyShellPkgs.out.packages ++ baseDevShellPkgs ++ [config.packages.zigrad-sdk-full-gpu];
        env =
          pyShellPkgs.out.env
          // sdkEnv
          // {
            CUDA_HOME = "${config.packages.cuda-redist.dev}";
            # NVRTC include paths for nix-compatible CUDA compilation
            NIX_GLIBC_INCLUDE = "${pkgs.stdenv.cc.libc.dev}/include";
            NIX_GCC_INCLUDE = "${pkgs.stdenv.cc.cc}/lib/gcc/${pkgs.stdenv.hostPlatform.config}/${pkgs.lib.getVersion pkgs.stdenv.cc.cc}/include";
          };
        shellHook =
          pjrtSelectHook
          + lspShadowHook
          + ''
            # zig is not happy about -fmacro-prefix-map
            unset NIX_CFLAGS_COMPILE
          '';
      };

      pure = pkgs.mkShellNoCC {
        packages = baseDevShellPkgs ++ [config.packages.zigrad-sdk-full-gpu];
        env = sdkEnv;
        shellHook = pjrtSelectHook + lspShadowHook;
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
        shellHook = pjrtSelectHook + lspShadowHook;
      };
    };
  };
}
