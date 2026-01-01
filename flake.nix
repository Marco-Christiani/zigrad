{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    nix-gl-host.url = "github:numtide/nix-gl-host";
    # this fork isnt exactly as correct but i hope its faster bc the mainline one is really slow like 20-30s startup
    #   i should fork and fix one of them or write my own idk but if this works and is faster then im happy.
    #   that being said, this seems like it may be bringing in gigs of deps (ironically, given the stated motivations)
    #   although i would need to actually check this to be confident in that idea.
    # nix-gl-host.url = "github:arilotter/nix-gl-host-rs";

    xla-src = {
      # local checkout layout
      # url = "path:./reference/xla";
      # For fully remote hermeticity, "" and run (run `nix flake lock`)
      # Get commit hash with: `git -C reference/xla/ checkout $XLA_TAG && git -C reference/xla/ rev-parse HEAD`
      url = "github:openxla/xla/913ae2eaa3cb88971003592a90959685a78c9e30";
      flake = false;
    };
  };

  outputs = { self, nixpkgs, nix-gl-host, xla-src }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" ];

      forAllSystems = f:
        builtins.listToAttrs (map (system: { name = system; value = f system; }) systems);

      cudaCfg = import ./nix/cuda.nix;

      colors = {
        yellow = "\\033[33m";
        reset = "\\033[0m";
      };

      mkFor = system:
        let
          pkgs = import nixpkgs {
            inherit system;
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

          targets = (import ./nix/targets.nix {
            inherit pkgs cudaPackages gccHost nixglhost src;
            cudaArchitectures = cudaCfg.cudaArchitectures;
          }).targets;

          cudaArchStr = pkgs.lib.concatStringsSep ";" cudaCfg.cudaArchitectures;

          pjrtCudaWheels = import ./nix/pjrt-cuda-wheels.nix;
          pjrtCudaBundle = pkgs.callPackage ./nix/pjrt-cuda-bundle.nix {
            wheelSources = pjrtCudaWheels;
          };
          pjrtCudaBundleDevel =
            pkgs.callPackage ./nix/pjrt-cuda-bundle.nix {
              wheelSources = pjrtCudaWheels;
              withHeaders = true;
            };

          xlaMlirStablehloSdk =
            pkgs.callPackage ./nix/xla-mlir-stablehlo-capi-sdk.nix {
              inherit xla-src;
            };
        in
        {
          # primary deliverable are devshells. their intended purpose is really just fast iteration and pinned
          #   toolchain with relaxed hermeticity requirements as needed for productivity.
          devShells = {
            # Minimal shell for M4 MLIR work (fast to enter)
            m4 = pkgs.mkShellNoCC {
              packages = with pkgs; [
                zig
                # MLIR/LLVM for M4 (matches JAX's LLVM 22.0.0git from Dec 28, 2025)
                llvmPackages_git.llvm
                llvmPackages_git.libllvm
              ];
            };

            default = pkgs.mkShellNoCC {
              packages = with pkgs; [
                go-task
                cmake
                gnumake

                zig
                zls

                cudaPackages.cudatoolkit
                cudaPackages.cuda_cudart
                gccHost
                stdenv.cc.cc.lib # provides libstdc++.so.6 for PJRT plugin

                # provides nixglhost binary
                nixglhost

                # MLIR/LLVM for M4 (matches JAX's LLVM 22.0.0git from Dec 28, 2025)
                llvmPackages_git.llvm
                llvmPackages_git.libllvm
              ];

              # pin the host compiler nvcc will use
              CC = "${gccHost}/bin/gcc";
              CXX = "${gccHost}/bin/g++";
              CUDACXX = "${cudaPackages.cudatoolkit}/bin/nvcc";

              # dev-time override for CUDA arches without editing nix:
              #   export CUDA_ARCHS="86;89"  (CMake list syntax)
              #   task conf
              shellHook = ''
                export CUDA_HOME=${cudaPackages.cudatoolkit}
                export CUDA_ARCHS="''${CUDA_ARCHS:-${cudaArchStr}}"
                export PJRT_CPU_PLUGIN_PATH=lib/pjrt_c_api_cpu_plugin.so

                if [[ ! -f $PJRT_CPU_PLUGIN_PATH ]]; then
                  # TODO: pulling the .so and possibly the pjrt header should be done in either nix or build.zig.zon
                  printf "${colors.yellow}WARNING:${colors.reset} PJRT_CPU_PLUGIN_PATH=$PJRT_CPU_PLUGIN_PATH does not exist. Leaving the env variable set but might need to pull this.${colors.reset}.\n"
                fi

                # Add libstdc++ to LD_LIBRARY_PATH for PJRT plugin
                export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:''${LD_LIBRARY_PATH:-}"

                echo "CUDA_HOME=$CUDA_HOME"
                echo "CUDA_ARCHS=$CUDA_ARCHS"
                echo "note: run GPU binaries via nixglhost (wrap the program):"
                echo "  nixglhost ./build/programs/main"
              '';
            };

            # convenience shell that DOES do global LD_LIBRARY_PATH injection 
            #   NOTE: discouraged by nix-gl-host docs. we keep it separate so default stays sane.
            impure-driver = pkgs.mkShellNoCC {
              inputsFrom = [ self.devShells.${system}.default ];
              shellHook = ''
                # nix-gl-host docs: -p is discouraged / footgun, this is for convenience only.
                export LD_LIBRARY_PATH="$(${nixglhost}/bin/nixglhost -p):''${LD_LIBRARY_PATH:-}"
                echo "LD_LIBRARY_PATH injected via nixglhost -p (convenience shell)."
              '';
            };

            xla-mlir = pkgs.mkShellNoCC {
              packages = with pkgs; [
                xlaMlirStablehloSdk
                cmake
                ninja
              ];

              shellHook = ''
                export XLA_CAPI_SDK=${xlaMlirStablehloSdk}
                export CPATH=$XLA_CAPI_SDK/include:$CPATH
                export LIBRARY_PATH=$XLA_CAPI_SDK/lib:$LIBRARY_PATH
                export LD_LIBRARY_PATH=$XLA_CAPI_SDK/lib:$LD_LIBRARY_PATH

                echo "Using XLA-derived MLIR+StableHLO C API SDK:"
                echo "  XLA_CAPI_SDK=$XLA_CAPI_SDK"
              '';
            };
          };

          # secondary deliverable are hermetic packages + explicit run wrappers (secondary bc we dont rly have a finished thing rn)
          packages = {
            # expose group explicitly for future extension
            example-cuda = targets.example-cuda.build;
            example-cuda-run = targets.example-cuda.run;

            gen-clangd = targets.editor.clangd;
            gen-nvim = targets.editor.nvim;
            # TODO: hermetic zig build/run targets
            # m1 = targets.m1.build;
            # m1 = targets.m1.run;

            pjrt-cuda-bundle = pjrtCudaBundle;

            # devel option includes headers in the bundle
            pjrt-cuda-bundle-devel = pjrtCudaBundleDevel;

            # hermetic MLIR / StableHLO C API SDK derived from XLA
            xla-mlir-stablehlo-capi-sdk = xlaMlirStablehloSdk;
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
          };
        };
    in
    {
      devShells = forAllSystems (s: (mkFor s).devShells);
      packages = forAllSystems (s: (mkFor s).packages);
      apps = forAllSystems (s: (mkFor s).apps);
      formatter = forAllSystems (system: let pkgs = import nixpkgs { inherit system; }; in pkgs.nixpkgs-fmt);
    };
}
