{
  pkgs,
  xlaSrc,
  llvmSrc,
  llvmRevision,
  withDebugSymbols,
  enableLto,
  extraCxxFlags,
  extraLdFlags,
}:
# This is the shared LLVM and MLIR toolchain for integrations that must load
#  into the same process. Its current revision and patch set are XLA-aligned.
pkgs.callPackage ../../packages/llvm.nix {
  inherit xlaSrc llvmSrc llvmRevision withDebugSymbols enableLto extraCxxFlags extraLdFlags;
  withNativeTuning = false;
}
