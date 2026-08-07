{
  pkgs,
  llvmSrc,
  llvmRevision,
  llvmVersion,
  withDebugSymbols,
  enableLto,
  extraCxxFlags,
  extraLdFlags,
}:
# This is the shared LLVM and MLIR toolchain for integrations that load into
#  the same process.
pkgs.callPackage ../../packages/llvm.nix {
  inherit llvmSrc llvmRevision llvmVersion withDebugSymbols enableLto extraCxxFlags extraLdFlags;
  withNativeTuning = false;
}
