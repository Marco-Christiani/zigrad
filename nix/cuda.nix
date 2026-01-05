{
  # - this is the project-wide CUDA arch policy
  # - keep it centralized so we never hardcode sm_XX in random places.
  #
  # CMake expects a list like: "86;89" or "86".
  # If you want multi-arch binaries, put multiple entries here.
  #
  # For dev convenience you can override at runtime via CUDA_ARCHS, see devShell.
  cudaArchitectures = ["86"];

  # prefer explicit host compiler for nvcc to avoid it discovering something else.
  gccHostAttr = "gcc13";

  # NOTE: pick the CUDA toolchain once. change here, not in random shells.
  cudaPackagesAttr = "cudaPackages_13";
}
