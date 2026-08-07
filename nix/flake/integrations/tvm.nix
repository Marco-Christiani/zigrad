{
  pkgs,
  cudaToolkit,
  cudaRuntime,
  gccHost,
  llvm,
  cudaArchitectures,
  withDebugSymbols,
  withNativeTuning,
  enableLto,
  extraCxxFlags,
  extraLdFlags,
  source,
}: let
  tvm = pkgs.callPackage ../../packages/tvm.nix {
    inherit
      cudaToolkit
      gccHost
      llvm
      withDebugSymbols
      withNativeTuning
      enableLto
      cudaArchitectures
      extraCxxFlags
      extraLdFlags
      ;
    inherit cudaRuntime;
    inherit (source) src;
    version = source.rev;
    cudaSupport = true;
  };
  tvmFullDev = pkgs.callPackage ../../packages/tvm.nix {
    inherit
      cudaToolkit
      gccHost
      llvm
      withDebugSymbols
      withNativeTuning
      enableLto
      cudaArchitectures
      extraCxxFlags
      extraLdFlags
      ;
    inherit cudaRuntime;
    inherit (source) src;
    version = source.rev;
    cudaSupport = true;
    withPythonBindings = true;
  };
  tvmCpu = pkgs.callPackage ../../packages/tvm.nix {
    inherit
      gccHost
      llvm
      withDebugSymbols
      withNativeTuning
      enableLto
      extraCxxFlags
      extraLdFlags
      ;
    inherit (source) src;
    version = source.rev;
    cudaSupport = false;
  };
in {
  inherit tvm tvmCpu tvmFullDev;
}
