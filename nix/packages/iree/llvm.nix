# Build LLVM, Clang, LLD, and MLIR from IREE's llvm-project fork.
#
# IREE carries patches that diverge from the XLA-pinned LLVM package. This
#  derivation follows IREE's LLVM and MLIR configuration for ABI compatibility.
{
  lib,
  stdenv,
  cmake,
  ninja,
  python3,
  perl,
  patchelf,
  zlib,
  zstd,
  lld,
  binutils,
  ireeLlvmSrc,
  ireeLlvmRevision,
  # Retain debug information in a RelWithDebInfo build.
  withDebugSymbols ? false,
  # Native tuning has limited effect on these compiler libraries.
  withNativeTuning ? false,
  enableLto ? false,
  extraCxxFlags ? [],
  extraLdFlags ? [],
}:
stdenv.mkDerivation {
  pname = "iree-llvm";
  version = "iree-llvm-${builtins.substring 0 7 ireeLlvmRevision}";

  # Build the required projects in one LLVM CMake graph.
  src = ireeLlvmSrc;

  strictDeps = true;
  dontStrip = withDebugSymbols;

  # Native LLVM and MLIR build tools load these libraries in the build sandbox.
  nativeBuildInputs = [
    cmake
    ninja
    python3
    perl
    patchelf
    lld
    binutils
    stdenv.cc.cc.lib
    zlib
    zstd
  ];

  buildInputs = [
    zlib
    zstd
  ];

  # The LLVM repository's CMake root is the `llvm` subdirectory.
  cmakeDir = "../llvm";

  cmakeFlags =
    [
      "-DCMAKE_BUILD_TYPE=${
        if withDebugSymbols
        then "RelWithDebInfo"
        else "Release"
      }"

      # Enable the projects required by the IREE compiler build.
      "-DLLVM_ENABLE_PROJECTS=mlir;clang;lld"

      # Retain host, AArch64, and CUDA code generation.
      "-DLLVM_TARGETS_TO_BUILD=X86;AArch64;NVPTX"

      # Link tools and libraries through `libLLVM.so`.
      "-DLLVM_BUILD_LLVM_DYLIB=ON"
      "-DLLVM_LINK_LLVM_DYLIB=ON"

      # Install the MLIR C library without changing IREE's tool-linking policy.
      "-DMLIR_BUILD_MLIR_C_DYLIB=ON"
      "-DMLIR_LINK_MLIR_DYLIB=OFF"

      # Exclude unused project outputs and optional system integrations.
      "-DLLVM_INCLUDE_TESTS=OFF"
      "-DLLVM_INCLUDE_EXAMPLES=OFF"
      "-DLLVM_INCLUDE_DOCS=OFF"
      "-DMLIR_INCLUDE_TESTS=OFF"
      "-DLLVM_ENABLE_UNWIND_TABLES=OFF"
      "-DLLVM_ENABLE_TERMINFO=OFF"
      "-DLLVM_ENABLE_LIBEDIT=OFF"
      "-DLLVM_ENABLE_LIBXML2=OFF"
      "-DLLVM_ENABLE_FFI=OFF"
      "-DLLVM_ENABLE_Z3_SOLVER=OFF"
      "-DLLVM_INCLUDE_GO_TESTS=OFF"

      "-DMLIR_ENABLE_BINDINGS_PYTHON=OFF"

      # Install build utilities consumed by the IREE compiler derivation.
      "-DLLVM_INSTALL_UTILS=ON"
      "-DLLVM_BUILD_UTILS=ON"

      "-DLLVM_USE_LINKER=lld"

      # Make build-tree tools resolve their Nix-provided libraries.
      "-DCMAKE_BUILD_RPATH=${lib.makeLibraryPath [stdenv.cc.cc.lib zlib zstd]}"
      "-DCMAKE_BUILD_RPATH_USE_ORIGIN=ON"
    ]
    ++ lib.optional enableLto "-DLLVM_ENABLE_LTO=Thin"
    ++ (let
      cxxFlags = (lib.optionals withNativeTuning ["-march=native" "-mtune=native"]) ++ extraCxxFlags;
    in
      lib.optional (cxxFlags != []) "-DCMAKE_CXX_FLAGS=${lib.concatStringsSep " " cxxFlags}")
    ++ lib.optional (extraLdFlags != []) "-DCMAKE_SHARED_LINKER_FLAGS=${lib.concatStringsSep " " extraLdFlags}";

  postInstall = ''
    set -euo pipefail

    rpath="\$ORIGIN:\$ORIGIN/../lib:${
      lib.makeLibraryPath [
        zlib
        zstd
        stdenv.cc.cc.lib
      ]
    }"

    for f in "$out/lib/"*.so*; do
      [ -f "$f" ] || continue
      patchelf --set-rpath "$rpath" "$f" || true
    done

    for f in "$out/bin/"*; do
      [ -x "$f" ] && [ -f "$f" ] || continue
      patchelf --set-rpath "$rpath" "$f" 2>/dev/null || true
    done

    # Nix fixup hooks expect unset variables to expand without failure.
    set +u
  '';

  meta = {
    description = "LLVM+Clang+LLD+MLIR built from IREE's fork, for use as BYO-LLVM in iree-compiler";
    license = lib.licenses.asl20;
  };
}
