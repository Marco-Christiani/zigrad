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
}: let
  ireeLlvm = pkgs.callPackage ../../packages/iree/llvm.nix {
    inherit ireeLlvmSrc ireeLlvmRevision withDebugSymbols enableLto extraCxxFlags extraLdFlags;
    withNativeTuning = false;
  };
  ireeCompiler = pkgs.callPackage ../../packages/iree/compiler.nix {
    inherit
      ireeSrc
      ireeRevision
      ireeStablehloSrc
      ireeFlatccSrc
      ireeBenchmarkSrc
      ireeLlvm
      withDebugSymbols
      enableLto
      extraCxxFlags
      extraLdFlags
      ;
    withNativeTuning = false;
  };
  ireeRuntime = pkgs.callPackage ../../packages/iree/runtime.nix {
    inherit
      ireeSrc
      ireeRevision
      ireeStablehloSrc
      ireeFlatccSrc
      ireeBenchmarkSrc
      ireeLlvm
      withDebugSymbols
      withNativeTuning
      enableLto
      extraCxxFlags
      extraLdFlags
      ;
  };
in {
  inherit ireeCompiler ireeLlvm ireeRuntime;
}
