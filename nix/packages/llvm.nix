# Builds the LLVM revision selected by Zigrad's dependency snapshot.
#
# StableHLO integration and TVM share this build and its ABI.
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
  libxml2,
  ncurses,
  libedit,
  libffi,
  lld,
  binutils,
  # Source inputs.
  llvmSrc,
  llvmRevision,
  llvmVersion,
  # Retain debug information in a RelWithDebInfo build.
  withDebugSymbols ? false,
  # Emit host-specific CPU instructions in LLVM libraries.
  #
  # This remains off by default because multiple integrations share the
  #  resulting libraries.
  withNativeTuning ? false,
  # Enable ThinLTO for LLVM.
  enableLto ? false,
  extraCxxFlags ? [],
  extraLdFlags ? [],
}:
stdenv.mkDerivation {
  pname = "llvm";
  version = "${llvmVersion}-${builtins.substring 0 7 llvmRevision}";

  src = llvmSrc;
  patches = [../patches/llvm-mathextras.patch];

  strictDeps = true;
  dontStrip = withDebugSymbols;

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
    libxml2
    ncurses
    libedit
    libffi
  ];

  cmakeFlags = let
    cxxFlags = (lib.optionals withNativeTuning ["-march=native" "-mtune=native"]) ++ extraCxxFlags;
    ldFlags = extraLdFlags;
  in
    [
      "-DCMAKE_BUILD_TYPE=${
        if withDebugSymbols
        then "RelWithDebInfo"
        else "Release"
      }"
      "-DBUILD_SHARED_LIBS=ON"
      "-DLLVM_ENABLE_PROJECTS=mlir;clang;lld;polly"
      # TVM uses NVPTX for CUDA code generation.
      "-DLLVM_TARGETS_TO_BUILD=host;NVPTX"
      "-DLLVM_INCLUDE_TESTS=OFF"
      "-DLLVM_INCLUDE_EXAMPLES=OFF"
      "-DLLVM_INCLUDE_DOCS=OFF"
      "-DMLIR_INCLUDE_TESTS=OFF"
      "-DMLIR_ENABLE_BINDINGS_PYTHON=OFF"
      "-DMLIR_BUILD_MLIR_C_DYLIB=ON"
      "-DLLVM_USE_LINKER=lld"
      "-DLLVM_INSTALL_UTILS=ON"
      "-DCMAKE_BUILD_RPATH=${lib.makeLibraryPath [stdenv.cc.cc.lib zlib zstd]}"
      "-DCMAKE_BUILD_RPATH_USE_ORIGIN=ON"
    ]
    ++ lib.optional enableLto "-DLLVM_ENABLE_LTO=Thin"
    ++ lib.optional (cxxFlags != []) "-DCMAKE_CXX_FLAGS=${lib.concatStringsSep " " cxxFlags}"
    ++ lib.optional (ldFlags != []) "-DCMAKE_SHARED_LINKER_FLAGS=${lib.concatStringsSep " " ldFlags}";

  cmakeDir = "../llvm";

  postInstall = ''
    # Supply runtime dependencies to installed libraries and tools.
    rpath="\$ORIGIN:\$ORIGIN/../lib:${
      lib.makeLibraryPath [
        zlib
        zstd
        libxml2
        ncurses
        libedit
        libffi
        stdenv.cc.cc.lib
      ]
    }"

    for f in "$out/lib/"*.so*; do
      [ -f "$f" ] || continue
      patchelf --set-rpath "$rpath" "$f" || true
    done

    for f in "$out/bin/"*; do
      [ -x "$f" ] || continue
      [ -f "$f" ] || continue
      patchelf --set-rpath "$rpath" "$f" 2>/dev/null || true
    done

    if [ ! -x "$out/bin/llvm-config" ]; then
      echo "WARNING: llvm-config not found in $out/bin"
    fi
  '';

  meta = {
    description = "LLVM selected by the Zigrad dependency snapshot";
    # TODO(release): license, etc.
    license = lib.licenses.asl20;
    platforms = lib.platforms.linux;
  };
}
