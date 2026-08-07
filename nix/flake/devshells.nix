# nix/flake/devshells.nix
#
# Development shells for fast iteration with relaxed hermeticity.
{inputs, ...}: {
  perSystem = {
    pkgs,
    config,
    system,
    cudaCfg,
    zigradBuildConfigurations,
    ...
  }: let
    inherit (inputs) uv2nix;

    cudaPackages = pkgs.${cudaCfg.cudaPackagesAttr};
    gccHost = pkgs.${cudaCfg.gccHostAttr};

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

    integrationEnv = package: let
      externalInputs = package.externalInputs;
      externalInputsStr = toString externalInputs;
      runtimeStr = toString externalInputs.runtime;
    in
      package.configuration.runtimeEnv
      // {
        ZG_EXTERNAL_SDK_ROOT = externalInputsStr;
        ZG_RUNTIME_SDK_ROOT = runtimeStr;
        ZG_ZIG_BUILD_ARGS = package.configuration.zigFeatureFlags;
        ZG_MIRAGE_RUNTIME_LIBRARY = "${config.packages.mirage}/lib/libmirage_runtime.so";
        ZG_NLOHMANN_JSON_INCLUDE_DIR = "${pkgs.lib.getDev pkgs.nlohmann_json}/include";
        PJRT_CPU_PLUGIN_PATH = "${runtimeStr}/runtime/xla/pjrt/c/pjrt_c_api_cpu_plugin.so";
        PJRT_GPU_PLUGIN_PATH = "${runtimeStr}/runtime/xla/pjrt/c/pjrt_c_api_gpu_plugin.so";

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
        packages = pyShellPkgs.out.packages ++ baseDevShellPkgs ++ [config.packages.zigrad-dev-cuda];
        env =
          pyShellPkgs.out.env
          // integrationEnv config.packages.zigrad-dev-cuda;
        shellHook =
          lspShadowHook
          + ''
            export LD_LIBRARY_PATH="$ZG_RUNTIME_SDK_ROOT/lib:$ZG_RUNTIME_SDK_ROOT/runtime/sys/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

            # zig is not happy about -fmacro-prefix-map
            unset NIX_CFLAGS_COMPILE
          '';
      };

      tvm-python = pkgs.mkShellNoCC {
        packages = pyShellPkgs.out.packages ++ baseDevShellPkgs ++ [zigradBuildConfigurations.dev-cuda-tvm-python.package];
        env =
          pyShellPkgs.out.env
          // integrationEnv zigradBuildConfigurations.dev-cuda-tvm-python.package;
        shellHook =
          lspShadowHook
          + ''
            export PYTHONPATH="$ZG_EXTERNAL_SDK_ROOT/python''${PYTHONPATH:+:$PYTHONPATH}"
            export LD_LIBRARY_PATH="$ZG_RUNTIME_SDK_ROOT/lib:$ZG_RUNTIME_SDK_ROOT/runtime/sys/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

            # zig is not happy about -fmacro-prefix-map
            unset NIX_CFLAGS_COMPILE
          '';
      };

      zig = pkgs.mkShellNoCC {
        packages = baseDevShellPkgs ++ [config.packages.zigrad-dev-cuda];
        env = integrationEnv config.packages.zigrad-dev-cuda;
        shellHook =
          lspShadowHook
          + ''
            export LD_LIBRARY_PATH="$ZG_RUNTIME_SDK_ROOT/lib:$ZG_RUNTIME_SDK_ROOT/runtime/sys/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
          '';
      };

      profiling = pkgs.mkShellNoCC {
        packages =
          pyShellPkgs.out.packages
          ++ baseDevShellPkgs
          ++ [
            config.packages.zigrad-dev-cuda
            cudaPackages.nsight_systems
            cudaPackages.nsight_compute
          ];
        env =
          pyShellPkgs.out.env
          // integrationEnv config.packages.zigrad-dev-cuda;
        shellHook =
          lspShadowHook
          + ''
            export LD_LIBRARY_PATH="$ZG_RUNTIME_SDK_ROOT/lib:$ZG_RUNTIME_SDK_ROOT/runtime/sys/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
          '';
      };
    };
  };
}
