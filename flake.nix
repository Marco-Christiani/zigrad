{
  description = "A very basic flake";

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
      in
    {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            zigpkg
            mkl
            # yes, we use mkl but this needs to be installed to provide the cblas wrapper
            openblas
          ];
        };
    }
  );
}
