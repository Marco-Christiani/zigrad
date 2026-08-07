{
  pkgs,
  ireeSrc,
  ireeLlvmSrc,
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
    inherit ireeLlvmSrc withDebugSymbols enableLto extraCxxFlags extraLdFlags;
    withNativeTuning = false;
  };
  ireeCompiler = pkgs.callPackage ../../packages/iree/compiler.nix {
    inherit
      ireeSrc
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
