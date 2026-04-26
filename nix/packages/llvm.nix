# nix/llvm.nix
#
# Builds LLVM 22 from XLA-pinned sources. Used by both the
# MLIR/StableHLO SDK and TVM to ensure they share the same LLVM version.
#
# Sources are provided as flake inputs (xlaSrc for patches, llvmSrc for LLVM tree).
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
  # Flake source inputs (replacing lockFile).
  xlaSrc,
  llvmSrc,
  # When true: build with RelWithDebInfo, retain DWARF, don't strip.
  # When false (default, production): Release, NDEBUG, stripped.
  withDebugSymbols ? false,
  # Emit native CPU instructions in LLVM's own libraries. Off by default
  #  even when buildCfg sets it true — LLVM ships portable libs for every
  #  consumer (TVM, our SDK), so non-portable codegen here would force
  #  consumers onto the same CPU family. Tooling (llc, opt) can still
  #  detect host features at runtime via -mcpu=native.
  withNativeTuning ? false,
  # Link-time optimization. LLVM with LTO is HUGE build cost (~2-3x) for
  #  small TVM/SDK gains since LLVM is mostly compiler infra, not hot path.
  #  Off by default. Wire opt-in.
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
    in [
      "-DCMAKE_BUILD_TYPE=${if withDebugSymbols then "RelWithDebInfo" else "Release"}"
      "-DBUILD_SHARED_LIBS=ON"
      "-DLLVM_ENABLE_PROJECTS=mlir;clang;polly"
      # Include NVPTX for TVM CUDA code generation
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
      # Patch RUNPATH for all shared libs and binaries
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

      # Also patch binaries (llvm-config, etc.)
      for f in "$out/bin/"*; do
        [ -x "$f" ] || continue
        [ -f "$f" ] || continue
        patchelf --set-rpath "$rpath" "$f" 2>/dev/null || true
      done

      # Ensure llvm-config is available
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
