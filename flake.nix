{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    # nix-gl-host.url = "github:numtide/nix-gl-host";
    # this fork isnt exactly as correct but i hope its faster bc the mainline one is really slow like 20-30s startup
    #   i should fork and fix one of them or write my own idk but if this works and is faster then im happy.
    #   that being said, this seems like it may be bringing in gigs of deps (ironically, given the stated motivations)
    #   although i would need to actually check this to be confident in that idea.
    nix-gl-host.url = "github:arilotter/nix-gl-host-rs";
  };

  outputs = { self, nixpkgs, nix-gl-host }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" ];

      forAllSystems = f:
        builtins.listToAttrs (map (system: { name = system; value = f system; }) systems);

      cudaCfg = import ./nix/cuda.nix;

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

          t = import ./nix/targets.nix {
            inherit pkgs cudaPackages gccHost nixglhost src;
            cudaArchitectures = cudaCfg.cudaArchitectures;
          };

          cudaArchStr = pkgs.lib.concatStringsSep ";" cudaCfg.cudaArchitectures;
        in
        {
          # primary deliverable are devshells. their intended purpose is really just fast iteration and pinned
          #   toolchain with relaxed hermeticity requirements as needed for productivity.
          devShells = {
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

                # provides nixglhost binary
                nixglhost
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
          };

          # secondary deliverable are hermetic packages + explicit run wrappers (secondary bc we dont rly have a finished thing rn)
          packages = {
            # expose group explicitly for future extension
            main = t.targets.main.build;
            main-run = t.targets.main.run;

            gen-clangd = t.targets.editor.clangd;
            gen-nvim = t.targets.editor.nvim;
          };

          apps = {
            main = {
              type = "app";
              program = "${t.targets.main.run}/bin/main";
            };
            gen-clangd = {
              type = "app";
              program = "${t.targets.editor.clangd}/bin/gen-clangd";
            };

            gen-nvim = {
              type = "app";
              program = "${t.targets.editor.nvim}/bin/gen-nvim";
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
