# nix/flake/devshells.nix
#
# Development shells for fast iteration with relaxed hermeticity.
{
  perSystem = {
    pkgs,
    config,
    cudaCfg,
    zigradBuildConfigurations,
    ...
  }: let
    externalSources = import ../external-sources.nix {inherit pkgs;};
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

    mlirCppExampleDev = config.packages.zigrad-example-mlir-cpp-dev;

    baseDevShellPkgs = with pkgs; [
      zig
      zls
      zon2nix
      go-task
      binutils
      patchelf
      git
      gccHost
      clang
      cmake
      ninja
      clangdWrapped
    ];

    lspShadowHook = ''
      export PATH="${clangdWrapped}/bin:$PATH"
    '';

    mlirCppExampleHook = ''
      export PATH="${mlirCppExampleDev}/bin:$PATH"
    '';

    tvmPython = pkgs.python312.withPackages (pythonPackages:
      with pythonPackages; [
        cloudpickle
        ml-dtypes
        numpy
        packaging
        psutil
        scipy
        tornado
        typing-extensions
        xgboost
      ]);

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
        ZG_MLIR_VIM_RT = "${externalSources.llvm.src}/mlir/utils/vim";
        ZG_LLVM_VIM_RT = "${externalSources.llvm.src}/llvm/utils/vim";
      };

    runtimeLibraryHook = package: let
      runtime = toString package.externalInputs.runtime;
      paths =
        [
          "${runtime}/lib"
          "${runtime}/runtime/sys/lib"
        ]
        ++ pkgs.lib.optional
        (pkgs.lib.elem "cuda-driver" package.configuration.resolved)
        "/run/opengl-driver/lib";
    in ''
      export LD_LIBRARY_PATH="${pkgs.lib.concatStringsSep ":" paths}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    '';
  in {
    devShells = {
      default = pkgs.mkShellNoCC {
        packages = baseDevShellPkgs ++ [config.packages.zigrad-dev-cuda];
        env = integrationEnv config.packages.zigrad-dev-cuda;
        shellHook =
          lspShadowHook
          + runtimeLibraryHook config.packages.zigrad-dev-cuda
          + ''
            # zig is not happy about -fmacro-prefix-map
            unset NIX_CFLAGS_COMPILE
          '';
      };

      tvm-python = pkgs.mkShellNoCC {
        packages = baseDevShellPkgs ++ [tvmPython zigradBuildConfigurations.dev-cuda-tvm-python.package];
        env = integrationEnv zigradBuildConfigurations.dev-cuda-tvm-python.package;
        shellHook =
          lspShadowHook
          + runtimeLibraryHook zigradBuildConfigurations.dev-cuda-tvm-python.package
          + ''
            export PYTHONPATH="$ZG_EXTERNAL_SDK_ROOT/python''${PYTHONPATH:+:$PYTHONPATH}"

            # zig is not happy about -fmacro-prefix-map
            unset NIX_CFLAGS_COMPILE
          '';
      };

      zig = pkgs.mkShellNoCC {
        packages = baseDevShellPkgs ++ [config.packages.zigrad-dev-cuda];
        env = integrationEnv config.packages.zigrad-dev-cuda;
        shellHook =
          lspShadowHook
          + runtimeLibraryHook config.packages.zigrad-dev-cuda;
      };

      profiling = pkgs.mkShellNoCC {
        packages =
          baseDevShellPkgs
          ++ [
            config.packages.zigrad-dev-cuda
            cudaPackages.nsight_systems
            cudaPackages.nsight_compute
          ];
        env = integrationEnv config.packages.zigrad-dev-cuda;
        shellHook =
          lspShadowHook
          + runtimeLibraryHook config.packages.zigrad-dev-cuda;
      };

      mlir-cpp = pkgs.mkShellNoCC {
        packages = baseDevShellPkgs ++ [mlirCppExampleDev config.packages.zigrad-dev-cuda];
        env = integrationEnv config.packages.zigrad-dev-cuda;
        shellHook =
          lspShadowHook
          + mlirCppExampleHook
          + runtimeLibraryHook config.packages.zigrad-dev-cuda;
      };
    };
  };
}
