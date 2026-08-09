{
  description = "Zigrad MNIST training example";

  inputs = {
    zigrad.url = "github:Marco-Christiani/zigrad/6e5ad1835dba8c41c179eebc3d7727310a8b0ad4";
    nixpkgs.follows = "zigrad/nixpkgs";
  };

  outputs = {
    self,
    nixpkgs,
    zigrad,
  }: let
    systems = ["x86_64-linux"];
    forAllSystems = nixpkgs.lib.genAttrs systems;
    contextFor = system: let
      pkgs = import nixpkgs {
        inherit system;
        overlays = [zigrad.overlays.default];
      };
    in {
      inherit pkgs;
      configuration = zigrad.packages.${system}.zigrad-xla-iree-cpu;
    };
  in {
    packages = forAllSystems (system: let
      context = contextFor system;
    in {
      default = zigrad.lib.mkZigApplication {
        inherit (context) pkgs configuration;
        src = self;
        pname = "zigrad-mnist";
        mainProgram = "mnist";
      };
    });

    devShells = forAllSystems (system: let
      context = contextFor system;
    in {
      default = zigrad.lib.mkDevShell {
        inherit (context) pkgs configuration;
      };
    });
  };
}
