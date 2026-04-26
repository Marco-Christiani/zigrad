# nix/iree-llvm.nix
#
# Builds LLVM+Clang+LLD+MLIR from IREE's fork of llvm-project (iree-org/llvm-project).
# IREE maintains patches on top of upstream LLVM; this derivation must use their fork
# rather than our XLA-pinned llvm because the two forks diverge in MLIR internals.
#
# Output layout mirrors the existing llvm.nix derivation:
#   $out/lib/cmake/{llvm,clang,lld,mlir}  — cmake config dirs consumed by iree-compiler.nix
#   $out/include/mlir-c                    — MLIR C API headers
#   $out/lib/libMLIR-C.so.*                — MLIR C DSO
#   $out/lib/libLLVM-*.so                  — LLVM shared library
#
# The key difference from llvm.nix: we use the iree-org fork and follow IREE's own
# llvm_config.cmake/mlir_config.cmake presets (BYO-LLVM path from build_tools/llvm/).
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
  # Flake source input: iree-org/llvm-project fork.
  ireeLlvmSrc,
  # When true: RelWithDebInfo, retain DWARF, don't strip.
  # When false (default, production): Release, NDEBUG, stripped.
  withDebugSymbols ? false,
}:
stdenv.mkDerivation {
  pname = "iree-llvm";
  version = "iree-llvm-${ireeLlvmSrc.shortRev or "unknown"}";

  # Build from the llvm/ subdirectory, same as byo_llvm.sh do_build_llvm + do_build_mlir.
  # We combine them into a single cmake invocation by adding mlir to LLVM_ENABLE_PROJECTS,
  # which produces all four cmake config dirs (llvm, clang, lld, mlir) in one install.
  src = ireeLlvmSrc;

  strictDeps = true;
  dontStrip = withDebugSymbols;

  # System libs appear in both lists: nativeBuildInputs so that native build
  # tools compiled during the cmake build (e.g. mlir-linalg-ods-yaml-gen,
  # llvm-tblgen) can find their shared-library deps at runtime inside the Nix
  # sandbox under strictDeps; buildInputs so they are available for linking
  # the installed output libraries and headers.
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

  # Point cmake at the llvm/ subdirectory (not the repo root).
  cmakeDir = "../llvm";

  cmakeFlags = [
    "-DCMAKE_BUILD_TYPE=${if withDebugSymbols then "RelWithDebInfo" else "Release"}"

    # Enable all needed sub-projects in a single build.
    # mlir is our addition to IREE's llvm_config.cmake (which only lists clang;lld).
    # Including mlir here avoids the separate MLIR build step from byo_llvm.sh.
    "-DLLVM_ENABLE_PROJECTS=mlir;clang;lld"

    # Architectures IREE needs: X86 (host codegen) + NVPTX (CUDA codegen).
    "-DLLVM_TARGETS_TO_BUILD=X86;NVPTX"

    # Build + link against libLLVM.so (matches IREE llvm_config.cmake).
    # Tools and libraries link against the dylib, reducing disk/link cost.
    "-DLLVM_BUILD_LLVM_DYLIB=ON"
    "-DLLVM_LINK_LLVM_DYLIB=ON"

    # Build libMLIR-C.so (needed by our existing Zig bindings).
    # IREE's mlir_config.cmake explicitly sets MLIR_LINK_MLIR_DYLIB=OFF for
    # tool-linking compatibility; MLIR_BUILD_MLIR_C_DYLIB is a separate knob.
    "-DMLIR_BUILD_MLIR_C_DYLIB=ON"
    "-DMLIR_LINK_MLIR_DYLIB=OFF"

    # Reduce size: disable docs, tests, go tests, unwind tables.
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

    # No Python bindings needed for the LLVM layer.
    "-DMLIR_ENABLE_BINDINGS_PYTHON=OFF"

    # Install utils (FileCheck, llvm-config, etc.) needed by iree-compiler.nix.
    "-DLLVM_INSTALL_UTILS=ON"
    "-DLLVM_BUILD_UTILS=ON"

    # Use lld for fast linking.
    "-DLLVM_USE_LINKER=lld"

    # Embed the Nix store rpath at build time so tools run from the build tree.
    "-DCMAKE_BUILD_RPATH=${lib.makeLibraryPath [stdenv.cc.cc.lib zlib zstd]}"
    "-DCMAKE_BUILD_RPATH_USE_ORIGIN=ON"
  ];

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

    # Restore -u to default so fixupPhase's strip-hook doesn't trip.
    set +u
  '';

  meta = {
    description = "LLVM+Clang+LLD+MLIR built from IREE's fork, for use as BYO-LLVM in iree-compiler";
    license = lib.licenses.asl20;
  };
}
