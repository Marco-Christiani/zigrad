{
  pkgs,
  lib,
  repoRoot,
  cudaToolkit,
  cudaRuntime,
  cudaVersion,
  mkCudaPackage,
  gccHost,
  cudaArchitectures,
  withDebugSymbols,
  withNativeTuning,
  enableLto,
  extraCxxFlags,
  extraLdFlags,
  source,
}: let
  mirageCudaHeaders = mkCudaPackage {
    componentNames = ["libcublas"];
    includeDevelopmentFiles = true;
    includeRuntimeFiles = false;
    nameSuffix = "-mirage-headers";
  };
  mirageCudaToolkit = pkgs.symlinkJoin {
    name = "cuda-mirage-toolkit-${cudaVersion}";
    paths = [
      cudaToolkit
      mirageCudaHeaders.dev
    ];
  };
  mirageRustLibs = pkgs.callPackage ../../packages/mirage-rust-libs.nix {
    inherit (source) src;
    sourceRoot = source.source_root;
    lockFile = repoRoot + /nix/locks/mirage-Cargo.lock;
  };
  mirage = pkgs.callPackage ../../packages/mirage.nix {
    inherit
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
    cudaToolkit = mirageCudaToolkit;
    inherit (source) src;
    sourceRoot = source.source_root;
    revision = source.rev;
  };
  mirageAdapter = pkgs.callPackage ../../packages/mirage-adapter.nix {
    inherit
      gccHost
      mirage
      withDebugSymbols
      enableLto
      extraCxxFlags
      extraLdFlags
      ;
    inherit cudaRuntime;
    cudaToolkit = mirageCudaToolkit;
    src = let
      fs = lib.fileset;
    in
      fs.toSource {
        root = repoRoot + /src/c/mirage/adapter;
        fileset = fs.unions [
          (repoRoot + /src/c/mirage/adapter/CMakeLists.txt)
          (repoRoot + /src/c/mirage/adapter/mirage.cc)
          (repoRoot + /src/c/mirage/adapter/include)
        ];
      };
  };
in {
  inherit mirage mirageAdapter mirageRustLibs;
}
