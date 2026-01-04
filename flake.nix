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
  outputs =
    {
      self,
      nixpkgs,
      nix-gl-host,
    }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
      ];

      forAllSystems =
        f:
        builtins.listToAttrs (
          map (system: {
            name = system;
            value = f system;
          }) systems
        );

      cudaCfg = import ./nix/cuda.nix;

      colors = {
        yellow = "\\033[33m";
        reset = "\\033[0m";
        green = "\\033[32m";
      };

      mkFor =
        system:
        let
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
            })
            targets
            ;

          lockFile = ./nix/lock.json;

          pjrtCudaBundle = pkgs.callPackage ./nix/pjrt-cuda-bundle.nix {
            inherit lockFile;
            withNvidiaHeaders = false;
          };

          pjrtCudaBundleDevel = pkgs.callPackage ./nix/pjrt-cuda-bundle.nix {
            inherit lockFile;
            withNvidiaHeaders = true;
          };

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
              pjrtCudaBundle
              xlaMlirStablehloCapiSdk
              # zigradMlirShim
            ];
          };
          sdkRoot = toString zigradExternalSdk;

          # Dev: ccache + devel (save more build artifacts + NVIDIA headers)
          zigradExternalSdkDevel = pkgs.symlinkJoin {
            name = "zigrad-external-sdk-devel";
            paths = [
              pjrtCudaBundle
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
          baseDevShellPkgs = with pkgs; [
            zig
            zls
            go-task
            binutils
            patchelf
          ];
        in
        {
          # devshells intended purpose is really just fast iteration and pinned toolchain with
          #   relaxed hermeticity requirements as needed for productivity.
          devShells = {
            default = pkgs.mkShellNoCC {
              packages = baseDevShellPkgs ++  [ zigradExternalSdkDevel ];
              ZG_EXTERNAL_SDK_ROOT = sdkRootDevel;
              PJRT_PLUGIN_PATH = "${sdkRootDevel}/runtime/jax_plugins/xla_cuda13/xla_cuda_plugin.so";

              shellHook = ''
                if [[ ! -f $PJRT_PLUGIN_PATH ]]; then
                  printf "${colors.yellow}[WARNING]${colors.reset} PJRT_PLUGIN_PATH=$PJRT_PLUGIN_PATH does not exist. \
                          Leaving the env variable set but you may need to materialize this.\n"
                fi
                printf "SDK path: ZG_EXTERNAL_SDK_ROOT=$ZG_EXTERNAL_SDK_ROOT"
              '';
            };

            # A purer target that uses the real derivations with devel=true, no extra copies, no ccache.
            #   Can sandbox.
            pure = pkgs.mkShellNoCC {
              packages = baseDevShellPkgs ++  [ zigradExternalSdk ];
              ZG_EXTERNAL_SDK_ROOT = sdkRoot;
              PJRT_PLUGIN_PATH = "${sdkRoot}/runtime/jax_plugins/xla_cuda13/xla_cuda_plugin.so";

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

            # Runtime bundle (PJRT + Vendor DSOs as self-contained closure)
            pjrt-cuda-runtime = pjrtCudaBundle;

            # Runtime bundle - Dev target: ccache + devel (NVIDIA Headers)
            pjrt-cuda-runtime-devel = pjrtCudaBundleDevel;

            # Compile-time SDK (PJRT headers + MLIR + StableHLO)
            xla-mlir-stablehlo-capi-sdk = xlaMlirStablehloCapiSdk;

            # Comptile-time SDK - Dev target: ccache + devel
            xla-mlir-stablehlo-capi-sdk-devel = xlaMlirStablehloCapiDevel;

            gen-clangd = targets.editor.clangd;
            gen-nvim = targets.editor.nvim;
            # TODO: hermetic zig build/run targets
            # m1 = targets.m1.build;
            # m1 = targets.m1.run;
          };

          apps = {
            example-cuda-target = {
              type = "app";
              program = "${targets.example-cuda.run}/bin/example-cuda";
            };

            # m1 = {
            #   # TODO: hermetic zig build/run targets
            # };

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
    in
    {
      devShells = forAllSystems (s: (mkFor s).devShells);
      packages = forAllSystems (s: (mkFor s).packages);
      apps = forAllSystems (s: (mkFor s).apps);
      formatter = forAllSystems (
        system:
        let
          pkgs = import nixpkgs { inherit system; };
        in
        pkgs.nixpkgs-fmt
      );
    };
}
