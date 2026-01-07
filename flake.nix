{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    nix-gl-host.url = "github:numtide/nix-gl-host";
    # this fork isnt exactly as correct but i hope its faster bc the mainline one is really slow like 20-30s startup
    #   i should fork and fix one of them or write my own idk but if this works and is faster then im happy.
    #   that being said, this seems like it may be bringing in gigs of deps (ironically, given the stated motivations)
    #   although i would need to actually check this to be confident in that idea.
    # nix-gl-host.url = "github:arilotter/nix-gl-host-rs";
  };
  outputs = {
    self,
    nixpkgs,
    nix-gl-host,
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

      nixglhost = nix-gl-host.packages.${system}.default;

      # note to self: keep builds from accidentally capturing ./build, downloaded junk, etc.
      src = pkgs.lib.cleanSource self;
      # shimSrc = pkgs.lib.cleanSource (self + "/shim");

      inherit
        (import ./nix/targets.nix {
          inherit
            pkgs
            cudaPackages
            gccHost
            nixglhost
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

      xlaMlirStablehloCapiSdk = pkgs.callPackage ./nix/xla-mlir-stablehlo-capi-sdk.nix {
        inherit lockFile;
      };

      xlaMlirStablehloCapiDevel = pkgs.callPackage ./nix/xla-mlir-stablehlo-capi-sdk.nix {
        inherit lockFile;
        stdenv = pkgs.ccacheStdenv;
        devel = true;
      };

      # Convenience aggregate.
      #   others are individually targetable mostly for development reasons
      zigradExternalSdk = pkgs.symlinkJoin {
        name = "zigrad-external-sdk";
        paths = [
          xlaPjrtPluginsCuda
          xlaMlirStablehloCapiSdk
          # zigradMlirShim
        ];
      };
      sdkRoot = toString zigradExternalSdk;

      # Dev: ccache + devel (save more build artifacts + NVIDIA headers)
      zigradExternalSdkDevel = pkgs.symlinkJoin {
        name = "zigrad-external-sdk-devel";
        paths = [
          xlaPjrtPluginsCudaDevel
          xlaMlirStablehloCapiDevel
          # zigradMlirShimDevel
        ];
      };
      sdkRootDevel = toString zigradExternalSdkDevel;

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
      # Bazel-built PJRT C API plugins from XLA
      xlaPjrtPlugins = pkgs.callPackage ./nix/xla-pjrt-runtime.nix {
        inherit lockFile;
        devel = false;
        cudaSupport = false;
        cudaPackages = null;
        persistentBazelOutputBase = false;
      };

      xlaPjrtPluginsCuda = pkgs.callPackage ./nix/xla-pjrt-runtime.nix {
        inherit lockFile cudaPackages;
        inherit (cudaCfg) cudaArchitectures;
        devel = false;
        cudaSupport = true;
        copyNcclNvshmem = true;
        copyCudaTools = true;
        copyLibdevice = true;
        # Keep current behavior: use cudaPackages.backendStdenv for CUDA builds.
        # Set false to reduce build closure size if Bazel uses its own CUDA repos.
        useCudaStdenv = true;
        persistentBazelOutputBase = false;
      };

      # Dev: ccache + devel.
      # TODO: Can flip cudaSupport=true once we plumb CUDA env/toolchain.
      xlaPjrtPluginsDevel = pkgs.callPackage ./nix/xla-pjrt-runtime.nix {
        inherit lockFile;
        stdenv = pkgs.ccacheStdenv;
        devel = true;

        # FIXME: CPU-only rn.
        cudaSupport = false;
        cudaPackages = null;

        # Bazel incremental cache outside store
        persistentBazelOutputBase = true;
      };

      # Dev: ccache + devel + CUDA.
      xlaPjrtPluginsCudaDevel = pkgs.callPackage ./nix/xla-pjrt-runtime.nix {
        inherit lockFile cudaPackages;
        inherit (cudaCfg) cudaArchitectures;
        stdenv = pkgs.ccacheStdenv;
        devel = true;
        cudaSupport = true;
        copyNcclNvshmem = true;
        copyCudaTools = true;
        copyLibdevice = true;
        # Keep current behavior: use cudaPackages.backendStdenv for CUDA builds.
        # Set false to reduce build closure size if Bazel uses its own CUDA repos.
        useCudaStdenv = true;
        persistentBazelOutputBase = true;
      };
      # ------------------------------------------------------------------
      baseDevShellPkgs = with pkgs; [
        zig
        zls
        go-task
        binutils
        patchelf
      ];
    in {
      # devshells intended purpose is really just fast iteration and pinned toolchain with
      #   relaxed hermeticity requirements as needed for productivity.
      devShells = {
        default = pkgs.mkShellNoCC {
          packages = baseDevShellPkgs ++ [zigradExternalSdkDevel];
          ZG_EXTERNAL_SDK_ROOT = sdkRootDevel;
          # PJRT_PLUGIN_PATH = "${sdkRootDevel}/runtime/jax_plugins/xla_cuda13/xla_cuda_plugin.so";
          PJRT_CPU_PLUGIN_PATH = "${sdkRootDevel}/runtime/xla/pjrt/c/pjrt_c_api_cpu_plugin.so";
          PJRT_GPU_PLUGIN_PATH = "${sdkRootDevel}/runtime/xla/pjrt/c/pjrt_c_api_gpu_plugin.so";

          shellHook = ''
            [[ -f "$PJRT_CPU_PLUGIN_PATH" ]]
            cpu_plugin_exists=$?

            [[ -f "$PJRT_GPU_PLUGIN_PATH" ]]
            gpu_plugin_exists=$?

            if (( cpu_plugin_exists != 0 )); then
              printf "%b[WARNING]%b PJRT_CPU_PLUGIN_PATH=%s does not exist. Leaving the env variable set but you may need to materialize this.\n" \
                "${colors.yellow}" "${colors.reset}" "$PJRT_CPU_PLUGIN_PATH"
            fi

            if (( gpu_plugin_exists != 0 )); then
              printf "%b[WARNING]%b PJRT_GPU_PLUGIN_PATH=%s does not exist. Leaving the env variable set but you may need to materialize this.\n" \
                "${colors.yellow}" "${colors.reset}" "$PJRT_GPU_PLUGIN_PATH"
            fi

            # Selection logic:
            # - If both plugins exist, GPU is preferred
            # - If only GPU exists, use GPU
            # - Otherwise fall back to CPU (warnings already emitted)
            if (( gpu_plugin_exists == 0 )); then
              if (( cpu_plugin_exists == 0 )); then
                printf "%b[INFO]%b Both CPU and GPU plugins exist. Selecting GPU plugin as the preferred option.\n" \
                  "${colors.yellow}" "${colors.reset}"
              fi
              PJRT_PLUGIN_PATH="$PJRT_GPU_PLUGIN_PATH"
            else
              PJRT_PLUGIN_PATH="$PJRT_CPU_PLUGIN_PATH"
            fi
            export PJRT_PLUGIN_PATH

            printf "Plugin path: PJRT_PLUGIN_PATH=%s\n" "$PJRT_PLUGIN_PATH"
            printf "SDK path: ZG_EXTERNAL_SDK_ROOT=%s\n" "$ZG_EXTERNAL_SDK_ROOT"
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
      };

      # secondary deliverable are hermetic packages + explicit run wrappers (secondary bc we dont rly have a finished thing rn)
      packages = {
        # Convenience aggregate and primary target.
        #   others are individually targetable mostly for development reasons
        zigrad-external-sdk = zigradExternalSdk;

        # Dev target: ccache + devel
        zigrad-external-sdk-devel = zigradExternalSdkDevel;

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

        # Compile-time SDK (PJRT headers + MLIR + StableHLO)
        xla-mlir-stablehlo-capi-sdk = xlaMlirStablehloCapiSdk;

        # Comptile-time SDK - Dev target: ccache + devel
        xla-mlir-stablehlo-capi-sdk-devel = xlaMlirStablehloCapiDevel;

        gen-clangd = targets.editor.clangd;
        gen-nvim = targets.editor.nvim;
        # TODO: hermetic zig build/run targets
        # m1 = targets.m1.build;
        m4 = targets.zigrad-m4.build;
      };

      apps = {
        example-cuda-target = {
          type = "app";
          program = "${targets.example-cuda.run}/bin/example-cuda";
        };

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
      };
    };
  in {
    devShells = forAllSystems (s: (mkFor s).devShells);
    packages = forAllSystems (s: (mkFor s).packages);
    apps = forAllSystems (s: (mkFor s).apps);
    formatter = forAllSystems (
      system: let
        pkgs = import nixpkgs {inherit system;};
      in
        pkgs.alejandra
    );
  };
}
