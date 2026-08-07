{
  pkgs,
  cudaCfg,
}: let
  gccHost = pkgs.${cudaCfg.gccHostAttr};
  cudaRedist = pkgs.callPackage ../../packages/cuda-redist.nix {
    inherit (cudaCfg) cudaVersion;
  };
in {
  inherit gccHost cudaRedist;

  # Build-time consumers require nvcc, headers, libraries, and stubs.
  cudaToolkit = cudaRedist.dev;
}
