# nix/llvm.nix
#
# Builds LLVM 22 from XLA-pinned sources (via lockFile). Used by both the
# MLIR/StableHLO SDK and TVM to ensure they share the same LLVM version.
{
  lib,
  stdenv,
  fetchurl,
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
  lockFile,
}: let
  lock = builtins.fromJSON (builtins.readFile lockFile);
  inherit (lock) pins;
  inherit (pins) xla llvm;

  xlaTar = fetchurl {
    url = xla.tarball_url;
    hash = xla.hash_sri;
  };
  llvmTar = fetchurl {
    inherit (llvm) urls;
    hash = llvm.hash_sri;
  };

  llvmPatches = ["build.patch" "mathextras.patch" "toolchains.patch" "zstd.patch" "lit_test.patch"];
  llvmIgnoredPatches = ["generated.patch"];

  xlaSrc = runCommand "xla-src-${builtins.substring 0 12 xla.commit}" {} ''
    mkdir -p $out
    tar -xzf ${xlaTar} -C $out --strip-components=1
  '';

  llvmSrc = runCommand "llvm-src-${builtins.substring 0 12 llvm.commit}" {nativeBuildInputs = [patch];} ''
    set -euo pipefail
    mkdir -p "$out"
    tar -xzf ${llvmTar} -C "$out" --strip-components=1
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
    version = "22.0git-${builtins.substring 0 12 llvm.commit}";

    src = llvmSrc;

    strictDeps = true;
    dontStrip = true;

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

    cmakeFlags = [
      "-DCMAKE_BUILD_TYPE=Release"
      "-DBUILD_SHARED_LIBS=ON"
      "-DLLVM_ENABLE_PROJECTS=mlir"
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
    ];

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
