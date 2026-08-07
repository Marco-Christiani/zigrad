{
  pkgs,
  cudaCfg,
}: let
  gccHost = pkgs.${cudaCfg.gccHostAttr};
  mkCudaPackage = options:
    pkgs.callPackage ../../packages/cuda-redist.nix (
      {inherit (cudaCfg) cudaVersion;}
      // options
    );
  cudaRuntime = mkCudaPackage {
    componentNames = [
      "cuda_cudart"
      "cuda_nvrtc"
      "libnvjitlink"
    ];
    includeSystemRuntime = true;
    nameSuffix = "-runtime";
  };
  cudaToolkitPackage = mkCudaPackage {
    componentNames = [
      "cuda_cccl"
      "cuda_cudart"
      "cuda_nvcc"
      "cuda_nvrtc"
      "libnvjitlink"
    ];
    includeDevelopmentFiles = true;
    includeRuntimeFiles = false;
    nameSuffix = "-toolkit";
  };
in {
  inherit gccHost cudaRuntime mkCudaPackage;

  # Build-time consumers require nvcc, headers, libraries, and stubs.
  cudaToolkit = cudaToolkitPackage.dev;
}
