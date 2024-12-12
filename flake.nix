# NOTE: we use mkl but openblas also needs to be installed to provide the cblas wrapper
{
  description = "Zigrad flake";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    zig.url = "github:mitchellh/zig-overlay";
  };

  outputs = { self, nixpkgs, zig, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
    let
      pkgs = import nixpkgs {
        inherit system;
        config = {
          allowUnfree = true;
        };
      };
      zigpkg = zig.packages.${system}."0.13.0";

      defaultShell = {
        "aarch64-darwin" = pkgs.mkShell {
          buildInputs = with pkgs; [
            zigpkg
          ];
        };
        "x86_64-linux" = pkgs.mkShell {
          buildInputs = with pkgs; [
            zigpkg
            mkl
            openblas
          ];
        };
        "aarch64-linux" = pkgs.mkShell {
          buildInputs = with pkgs; [
            zigpkg
            openblas
          ];
        };
      };
      in
    {
        devShells.default = defaultShell.${system};
    }
  );
}
