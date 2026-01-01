# nix/xla-mlir-stablehlo-capi-sdk.nix
# Keeps artifacts unstripped
#
# devel = false (default)
#   version: xla-$xla.commit[0:12]
#   copy only:
#     - MLIR C API headers
#     - StableHLO C API headers
#     - PJRT headers
#     - libMLIR-C.so*
#     - libMLIRCAPI*.a
#     - libStablehloCAPI.a
# devel = true, additionally copy:
#   version: xla-$xla.commit[0:12]-devel
#   additionally copy:
#    - all MLIR + StableHLO libs from build dirs
#    - optional CMake configs (lib/cmake/*)
{ lib
, stdenv
, fetchurl
, runCommand
, cmake
, ninja
, python3
, perl
, unzip
, patch
, zlib
, zstd
, libxml2
, ncurses
, libedit
, libffi
, lockFile
, devel ? false
, enableCcache ? false
, ccache ? null
}:

let
  lock = builtins.fromJSON (builtins.readFile lockFile);
  pins = lock.pins;

  xla = pins.xla;
  llvm = pins.llvm;
  stablehlo = pins.stablehlo;

  xlaTar = fetchurl {
    url  = xla.tarball_url;
    hash = xla.hash_sri;
  };

  llvmTar = fetchurl {
    urls = llvm.urls;
    hash = llvm.hash_sri;
  };

  stablehloZip = fetchurl {
    urls = stablehlo.urls;
    hash = stablehlo.hash_sri;
  };

  llvmPatches = [
    "build.patch"
    "mathextras.patch"
    "toolchains.patch"
    "zstd.patch"
    "lit_test.patch"
  ];

  llvmIgnoredPatches = [
    "generated.patch"
  ];

  stablehloPatches = [
    "temporary.patch"
  ];

  xlaSrc = runCommand "xla-src-${builtins.substring 0 12 xla.commit}" {} ''
    mkdir -p $out
    tar -xzf ${xlaTar} -C $out --strip-components=1
  '';

  llvmSrc = runCommand "llvm-src-${builtins.substring 0 12 llvm.commit}"
    { nativeBuildInputs = [ patch ]; }
    ''
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
          *)
            echo "ERROR: New or unexpected LLVM patch detected: $p" >&2
            echo "       Please audit and update llvmPatches / llvmIgnoredPatches." >&2
            exit 1
            ;;
        esac
      done

      for p in ${lib.concatStringsSep " " llvmPatches}; do
        echo "[llvm] Applying $p"
        patch -p1 < "${xlaSrc}/third_party/llvm/$p"
      done
    '';

  stablehloSrc = runCommand "stablehlo-src-${builtins.substring 0 12 stablehlo.commit}"
    { nativeBuildInputs = [ unzip patch ]; }
    ''
      set -euo pipefail

      mkdir -p "$out"
      unzip -q ${stablehloZip} -d "$out"
      mv "$out"/*/* "$out"/
      cd "$out"

      echo "[stablehlo] Verifying patch set"

      expected_patches="${lib.concatStringsSep " " stablehloPatches}"
      actual_patches="$(cd ${xlaSrc}/third_party/stablehlo && ls *.patch | tr '\n' ' ')"

      for p in $actual_patches; do
        case " $expected_patches " in
          *" $p "*) ;;
          *)
            echo "ERROR: New or unexpected StableHLO patch detected: $p" >&2
            echo "       Please audit and update stablehloPatches." >&2
            exit 1
            ;;
        esac
      done

      for p in $expected_patches; do
        echo "[stablehlo] Applying $p"
        patch -p1 < "${xlaSrc}/third_party/stablehlo/$p"
      done
    '';
in

stdenv.mkDerivation {
  pname = "xla-mlir-stablehlo-capi-sdk";
  version =
    "xla-${builtins.substring 0 12 xla.commit}"
    + lib.optionalString devel "-devel";


  strictDeps = true;
  dontUnpack = true;
  dontConfigure = true;
  dontStrip = true;

  nativeBuildInputs = [ cmake ninja python3 perl ccache ];
  buildInputs = [ zlib zstd libxml2 ncurses libedit libffi ];

  buildPhase = ''
    set -euo pipefail

    # HACK: dev only.
    # CCACHE_DIR=$HOME/.cache/ccache nix build .#xla-mlir-stablehlo-capi-sdk \
    #   --option sandbox false \
    #   --impure \
    #   --override-input enableCcache true

    if [ "${lib.boolToString enableCcache}" = "true" ]; then
      export CCACHE_DIR="''${CCACHE_DIR:-/var/tmp/ccache}"
      export CCACHE_BASEDIR="$PWD"
      export CCACHE_COMPRESS=1
      export CCACHE_SLOPPINESS=time_macros
      echo "****************CCACHE ENABLED - IMPURE****************" >&2
    fi

    cmake_flags=(
      -DCMAKE_BUILD_TYPE=Release
      -DLLVM_ENABLE_PROJECTS=mlir
      -DLLVM_TARGETS_TO_BUILD=host
      -DLLVM_INCLUDE_TESTS=OFF
      -DLLVM_INCLUDE_EXAMPLES=OFF
      -DLLVM_INCLUDE_DOCS=OFF
      -DMLIR_INCLUDE_TESTS=OFF
      -DMLIR_ENABLE_BINDINGS_PYTHON=OFF
      -DMLIR_BUILD_MLIR_C_DYLIB=ON
    )

    if [ "${lib.boolToString enableCcache}" = "true" ]; then
      cmake_flags+=(
        -DCMAKE_C_COMPILER_LAUNCHER=ccache
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
        -DCMAKE_ASM_COMPILER=gcc
      )
    fi

    mkdir -p llvm-build
    cmake -S ${llvmSrc}/llvm -B llvm-build -G Ninja "''${cmake_flags[@]}"


    # Tools StableHLO headers generation might expect (safe even if unused later, tbd whats needed still)
    cmake --build llvm-build --target llvm-tblgen mlir-tblgen mlir-pdll
    cmake --build llvm-build --target MLIR-C

    mkdir -p stablehlo-build
    cmake -S ${stablehloSrc} -B stablehlo-build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DMLIR_DIR="$PWD/llvm-build/lib/cmake/mlir" \
      -DLLVM_DIR="$PWD/llvm-build/lib/cmake/llvm"

    ninja -C stablehlo-build StablehloCAPI

    # Build MLIR C API libraries (what we actually want)
    cmake --build llvm-build --target \
      MLIRCAPIIR \
      MLIRCAPIArith \
      MLIRCAPIMath \
      MLIRCAPISCF \
      MLIRCAPITransforms \
      MLIRCAPIFunc \
      MLIRCAPITensor
  '';

  installPhase = ''
    log() { echo "[xla-mlir-stablehlo-capi-sdk] $*" >&2; }

    copy_headers() {
      local src="$1"
      local dst="$2"
      mkdir -p "$dst"
      cp -r "$src"/* "$dst/"
    }

    copy_libs_matching() {
      local root="$1"
      shift
      [ -d "$root" ] || return 0
      find "$root" -type f \( "$@" \) -print -exec cp -v {} "$out/lib/" \;
    }

    set -euo pipefail

    mkdir -p "$out/include" "$out/lib"

    # Devel path: copy everything
    if [ "${lib.boolToString devel}" = "true" ]; then
      log "Devel mode enabled: copying ALL build artifacts"

      # Copy everything we built, verbatim, for inspection/debugging

      mkdir -p "$out/include/llvm" "$out/include/stablehlo"

      cp -r "${llvmSrc}/mlir/include" "$out/include/llvm/"
      cp -r "${stablehloSrc}/stablehlo" "$out/include/stablehlo/"

      cp -r "${xlaSrc}/xla/pjrt" "$out/include/xla/"

      # Copy all libs produced by LLVM + StableHLO builds
      find llvm-build stablehlo-build -type f \
        \( -name "*.a" -o -name "*.so*" \) \
        -print -exec cp -v {} "$out/lib/" \;

      log "Devel SDK install complete (no validation performed)"
      exit 0
    fi

    # Minimal path: copy specific things

    log "Installing headers (curated)"

    # MLIR C API headers
    copy_headers "${llvmSrc}/mlir/include/mlir-c" "$out/include/mlir-c"

    # StableHLO C API headers
    copy_headers "${stablehloSrc}/stablehlo/integrations/c" \
                "$out/include/stablehlo/integrations/c"

    # PJRT C API headers
    mkdir -p "$out/include/xla/pjrt/c"
    cp -v "${xlaSrc}/xla/pjrt/c/"*.h "$out/include/xla/pjrt/c/"

    log "Installing libraries (curated)"

    # StableHLO C API
    copy_libs_matching stablehlo-build \
      -name "libStablehloCAPI*.a" -o -name "libStablehloCAPI*.so*"

    # MLIR C API + shared MLIR-C
    copy_libs_matching llvm-build \
      -name "libMLIR-C.so*" \
      -o -name "libMLIRCAPI*.a"

    # Ensure linker-visible MLIR-C name
    mlir_c_so="$(ls -1 "$out/lib/libMLIR-C.so."* 2>/dev/null | head -n1 || true)"
    if [ -n "$mlir_c_so" ]; then
      ln -sfn "$(basename "$mlir_c_so")" "$out/lib/libMLIR-C.so"
    fi

    log "Verifying outputs"

    if ! find "$out/lib" -name "libMLIR-C.so" | grep -q .; then
      log "ERROR: libMLIR-C.so missing"
      exit 1
    fi

    if ! find "$out/lib" -name "libStablehloCAPI*" | grep -q .; then
      log "ERROR: StablehloCAPI missing"
      exit 1
    fi

    log "SDK installation complete"
  '';

  meta = {
    description = "MLIR + StableHLO + PJRT C API SDK derived from JAX->XLA lock.json";
    license = lib.licenses.asl20;
  };
}

