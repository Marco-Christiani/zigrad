{
  pkgs,
  ireeSrc,
  ireeRevision,
  ireeLlvmSrc,
  ireeLlvmRevision,
  ireeStablehloSrc,
  ireeFlatccSrc,
  ireeBenchmarkSrc,
  withDebugSymbols,
  withNativeTuning,
  enableLto,
  extraCxxFlags,
  extraLdFlags,
  cudaToolkit,
}: let
  ireeLlvm = pkgs.callPackage ../../packages/iree/llvm.nix {
    inherit ireeLlvmSrc ireeLlvmRevision withDebugSymbols enableLto extraCxxFlags extraLdFlags;
    withNativeTuning = false;
  };
  common = {
    inherit
      ireeSrc
      ireeRevision
      ireeStablehloSrc
      ireeFlatccSrc
      ireeBenchmarkSrc
      withDebugSymbols
      enableLto
      extraCxxFlags
      extraLdFlags
      ;
  };
  ireeCompilerCpu = pkgs.callPackage ../../packages/iree/compiler.nix (common
    // {
      inherit ireeLlvm;
      withNativeTuning = false;
      targetBackends = ["llvm-cpu"];
    });
  ireeCompilerCuda = pkgs.callPackage ../../packages/iree/compiler.nix (common
    // {
      inherit ireeLlvm cudaToolkit;
      withNativeTuning = false;
      targetBackends = ["cuda"];
    });
  ireeRuntimeCpu = pkgs.callPackage ../../packages/iree/runtime.nix (common
    // {
      inherit withNativeTuning;
      drivers = ["local-sync" "local-task"];
    });
  ireeRuntimeCuda = pkgs.callPackage ../../packages/iree/runtime.nix (common
    // {
      inherit withNativeTuning cudaToolkit;
      drivers = ["cuda"];
    });
  mkRuntimeCpuFor = targetPkgs:
    targetPkgs.callPackage ../../packages/iree/runtime.nix (common
      // {
        ireeHostTools = ireeRuntimeCpu.hostTools;
        withNativeTuning = false;
        drivers = ["local-sync"];
      });
in {
  inherit ireeCompilerCpu ireeCompilerCuda ireeLlvm ireeRuntimeCpu ireeRuntimeCuda mkRuntimeCpuFor;
}
