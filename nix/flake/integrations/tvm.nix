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
    cudaSupport = false;
  };
in {
  inherit tvm tvmCpu tvmFullDev;
}
