# Builds LLVM 22 from XLA-pinned sources.
#
# StableHLO integration and TVM share this build and its ABI.
{
  lib,
  stdenv,
  runCommand,
  cmake,
  ninja,
  python3,
  perl,
  patch,
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
  xlaSrc,
  llvmSrc,
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
}: let
  llvmPatches = ["build.patch" "mathextras.patch" "toolchains.patch" "zstd.patch" "lit_test.patch"];
  llvmIgnoredPatches = ["generated.patch"];

  patchedLlvmSrc = runCommand "llvm-src-patched" {nativeBuildInputs = [patch];} ''
    set -euo pipefail
    cp -r ${llvmSrc} "$out"
    chmod -R u+w "$out"
    cd "$out"

    echo "[llvm] Verifying patch set"
    expected_patches="${lib.concatStringsSep " " llvmPatches} ${lib.concatStringsSep " " llvmIgnoredPatches}"
    actual_patches="$(cd ${xlaSrc}/third_party/llvm && ls *.patch | tr '\n' ' ')"
    for p in $actual_patches; do
      case " $expected_patches " in
        *" $p "*) ;;
        *) echo "ERROR: New or unexpected LLVM patch detected: $p" >&2; exit 1;;
      esac
    done

    for p in ${lib.concatStringsSep " " llvmPatches}; do
      echo "[llvm] Applying $p"
      patch -p1 < "${xlaSrc}/third_party/llvm/$p"
    done
  '';
in
  stdenv.mkDerivation {
    pname = "llvm";
    version = "22.0git-${llvmSrc.shortRev or "unknown"}";

    src = patchedLlvmSrc;

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
      description = "LLVM 22 built from XLA-pinned sources";
      license = lib.licenses.asl20;
      platforms = lib.platforms.linux;
    };
  }
