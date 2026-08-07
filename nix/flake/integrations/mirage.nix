{
  pkgs,
  lib,
  cudaToolkit,
  cudaRuntime,
  gccHost,
  cudaArchitectures,
  withDebugSymbols,
  withNativeTuning,
  enableLto,
  extraCxxFlags,
  extraLdFlags,
}: let
  mirageRevision = "ffe38dff251017cdddeb1891cc808b43e7f51f25";
  mirageSourceRoot = "mirage-${mirageRevision}";
  mirageSrc = pkgs.fetchurl {
    name = "mirage-${mirageRevision}.tar.gz";
    url = "https://github.com/mirage-project/mirage/archive/${mirageRevision}.tar.gz";
    hash = "sha256-1AXdzUkXUmW3tqsEBYgF+GlMrDuKsK7oYL3Y/mf22C0=";
  };
  mirageRustLibs = pkgs.callPackage ../../packages/mirage-rust-libs.nix {
    src = mirageSrc;
    sourceRoot = mirageSourceRoot;
    lockFile = ../../locks/mirage-Cargo.lock;
  };
  mirage = pkgs.callPackage ../../packages/mirage.nix {
    inherit
      cudaToolkit
      gccHost
      mirageRustLibs
      cudaArchitectures
      withDebugSymbols
      withNativeTuning
      enableLto
      extraCxxFlags
      extraLdFlags
      ;
    inherit cudaRuntime;
    src = mirageSrc;
    sourceRoot = mirageSourceRoot;
    revision = mirageRevision;
  };
  mirageAdapter = pkgs.callPackage ../../packages/mirage-adapter.nix {
    inherit
      cudaToolkit
      gccHost
      mirage
      withDebugSymbols
      enableLto
      extraCxxFlags
      extraLdFlags
      ;
    inherit cudaRuntime;
    src = let
      fs = lib.fileset;
    in
      fs.toSource {
        root = ../../../src/c/mirage/adapter;
        fileset = fs.unions [
          ../../../src/c/mirage/adapter/CMakeLists.txt
          ../../../src/c/mirage/adapter/mirage.cc
          ../../../src/c/mirage/adapter/include
        ];
      };
  };
in {
  inherit mirage mirageAdapter mirageRustLibs;
}
