{
  pkgs,
  cudaCfg,
  cudaRuntime,
  mkCudaPackage,
  cudaArchitectures,
  xlaSrc,
  xlaRevision,
  stablehloSrc,
  llvm,
  withDebugSymbols,
  withNativeTuning,
  enableLto,
  extraCxxFlags,
  extraLdFlags,
  extraBazelFlags,
}: let
  xlaCudaRuntime = mkCudaPackage {
    componentNames = [
      "cuda_cupti"
      "cuda_nvcc"
      "cudnn"
      "libcublas"
      "libcufft"
      "libcusparse"
      "libnvshmem"
      "nccl"
    ];
    componentDependencies = [cudaRuntime];
    nameSuffix = "-xla-runtime";
  };
  # PJRT + XLA FFI headers are a pure source copy.
  pjrtHeaders = pkgs.runCommand "pjrt-xla-ffi-headers" {} ''
    mkdir -p $out/include/xla/pjrt/c $out/include/xla/ffi/api
    cp ${xlaSrc}/xla/pjrt/c/*.h $out/include/xla/pjrt/c/
    cp ${xlaSrc}/xla/ffi/api/*.h $out/include/xla/ffi/api/
  '';

  # XLA proto schemas back the protobuf decode tool.
  xlaProtos = pkgs.runCommand "xla-protos" {} ''
    mkdir -p $out/proto/xla/service
    cp ${xlaSrc}/xla/xla.proto $out/proto/xla/
    cp ${xlaSrc}/xla/xla_data.proto $out/proto/xla/
    cp ${xlaSrc}/xla/service/hlo.proto $out/proto/xla/service/
    cp ${xlaSrc}/xla/service/metrics.proto $out/proto/xla/service/
  '';

  xlaMlirStablehloCapiSdk = pkgs.callPackage ../../packages/xla-mlir-stablehlo-capi-sdk.nix {
    inherit xlaSrc xlaRevision stablehloSrc llvm withDebugSymbols enableLto extraCxxFlags extraLdFlags;
    withNativeTuning = false;
  };

  xlaPjrtPlugins = pkgs.callPackage ../../packages/xla-pjrt-runtime-bazel.nix {
    inherit xlaSrc xlaRevision withDebugSymbols enableLto extraBazelFlags;
    cudaSupport = false;
    cpuMathLibrary = "onednn";
    cpuNativeTuning = withNativeTuning;
    depsHash = "sha256-7sTZkp4pWDJz+ngf6vyS/Okn1tWyLHXIkIr5Ym/uUWA=";
  };

  xlaPjrtPluginsCuda = pkgs.callPackage ../../packages/xla-pjrt-runtime-bazel.nix {
    inherit xlaSrc xlaRevision withDebugSymbols enableLto extraBazelFlags cudaArchitectures;
    inherit (cudaCfg) cudaVersion;
    cudaSupport = true;
    cpuMathLibrary = "onednn-thunk";
    cpuNativeTuning = withNativeTuning;
    depsHash = "sha256-+ZFReBX1+BhW58gXTFvt66g8kLuihIwOIo3Ufew0Uac=";
  };
in {
  inherit
    llvm
    pjrtHeaders
    xlaMlirStablehloCapiSdk
    xlaPjrtPlugins
    xlaPjrtPluginsCuda
    xlaProtos
    xlaCudaRuntime
    ;
}
