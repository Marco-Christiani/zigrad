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
, patchelf
, zlib
, zstd
, libxml2
, ncurses
, libedit
, libffi
, lockFile
, devel ? false
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

  nativeBuildInputs = [ cmake ninja python3 perl patchelf ];
  buildInputs = [ zlib zstd libxml2 ncurses libedit libffi ];

  buildPhase = ''
    set -euo pipefail

    cmake_flags=(
      -DCMAKE_BUILD_TYPE=Release
      -DCMAKE_BUILD_RPATH=\$ORIGIN/../lib
      -DCMAKE_INSTALL_RPATH=\$ORIGIN
      -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON
      -DLLVM_ENABLE_PROJECTS=mlir
      -DLLVM_TARGETS_TO_BUILD=host
      -DLLVM_INCLUDE_TESTS=OFF
      -DLLVM_INCLUDE_EXAMPLES=OFF
      -DLLVM_INCLUDE_DOCS=OFF
      -DMLIR_INCLUDE_TESTS=OFF
      -DMLIR_ENABLE_BINDINGS_PYTHON=OFF
      -DMLIR_BUILD_MLIR_C_DYLIB=ON
    )

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

    # Devel path: copy everything (but keep the curated include layout as a superset)
    if [ "${lib.boolToString devel}" = "true" ]; then
      log "Devel mode enabled: copying ALL build artifacts (curated layout + full dumps)"

      # 1) Install curated headers so a single SDK root works for Zig @cImport
      copy_headers "${llvmSrc}/mlir/include/mlir-c" "$out/include/mlir-c"

      # MLIR generated + source headers required by mlir-c/*
      if [ -d llvm-build/include/mlir ]; then
        mkdir -p "$out/include/mlir"
        cp -r llvm-build/include/mlir/* "$out/include/mlir/"
      fi
      if [ -d llvm-build/tools/mlir/include/mlir ]; then
        mkdir -p "$out/include/mlir"
        cp -r llvm-build/tools/mlir/include/mlir/* "$out/include/mlir/" || true
      fi
      if [ -d "${llvmSrc}/mlir/include/mlir" ]; then
        mkdir -p "$out/include/mlir"
        cp -r "${llvmSrc}/mlir/include/mlir/"* "$out/include/mlir/" || true
      fi

      copy_headers "${stablehloSrc}/stablehlo/integrations/c" \
                   "$out/include/stablehlo/integrations/c"

      mkdir -p "$out/include/xla/pjrt/c"
      cp -v "${xlaSrc}/xla/pjrt/c/"*.h "$out/include/xla/pjrt/c/"

      # 2) Dump full source headers under a separate namespace for inspection
      # Dereference symlinks so we never write through to /nix/store
      log "Devel: resetting _full dump dir"
      rm -rf "$out/include/_full"
      mkdir -p "$out/include/_full/llvm/include" "$out/include/_full/llvm/mlir_include" \
              "$out/include/_full/stablehlo" "$out/include/_full/xla"

      log "Devel: dumping full LLVM/MLIR headers (dereferenced)"
      cp -rL "${llvmSrc}/llvm/include/"* "$out/include/_full/llvm/include/"
      cp -rL "${llvmSrc}/mlir/include/"* "$out/include/_full/llvm/mlir_include/"

      log "Devel: dumping full StableHLO + XLA PJRT headers (dereferenced)"
      cp -rL "${stablehloSrc}/stablehlo" "$out/include/_full/stablehlo/"
      cp -rL "${xlaSrc}/xla/pjrt" "$out/include/_full/xla/"

      # 3) Copy all libs produced by LLVM + StableHLO builds
      find llvm-build stablehlo-build -type f \
        \( -name "*.a" -o -name "*.so*" \) \
        -print -exec cp -v {} "$out/lib/" \;

      # Ensure linker-visible MLIR-C name exists if a versioned .so was produced
      mlir_c_so="$(ls -1 "$out/lib/libMLIR-C.so."* 2>/dev/null | head -n1 || true)"
      if [ -n "$mlir_c_so" ]; then
        ln -sfn "$(basename "$mlir_c_so")" "$out/lib/libMLIR-C.so"
      fi

      log "Devel SDK install complete (no validation performed)"
      exit 0
    fi

    # Minimal path: copy specific things

    log "Installing headers (curated)"

    # MLIR C API headers
    copy_headers "${llvmSrc}/mlir/include/mlir-c" "$out/include/mlir-c"

    # MLIR generated C-API include files.
    # Needed because mlir-c/*.h includes files like:
    #   #include 'mlir/Transforms/Transforms.capi.h.inc'
    # which are generated into the build include tree.
    if [ -d llvm-build/include/mlir ]; then
      mkdir -p "$out/include/mlir"
      cp -r llvm-build/include/mlir/* "$out/include/mlir/"
    fi
    if [ -d llvm-build/tools/mlir/include/mlir ]; then
      mkdir -p "$out/include/mlir"
      cp -r llvm-build/tools/mlir/include/mlir/* "$out/include/mlir/" || true
    fi
    # copy more just to be sure we have full ctx although these are probably all the pre-gen counterparts and above are post-gen.
    if [ -d "${llvmSrc}/mlir/include/mlir" ]; then
      mkdir -p "$out/include/mlir"
      cp -r "${llvmSrc}/mlir/include/mlir/"* "$out/include/mlir/" || true
    fi

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

    # StableHLO dialect implementation (needed by StablehloCAPI)
    copy_libs_matching stablehlo-build \
      -name "libStablehloOps.a"

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

    log "Fixing RPATHs"
    rpath="\$ORIGIN:\$ORIGIN/../runtime/sys/lib:${lib.makeLibraryPath [ zlib zstd libxml2 ncurses libedit libffi stdenv.cc.cc.lib ]}"
    for f in "$out/lib/libMLIR-C.so."*; do
      if [ -f "$f" ]; then
        patchelf --set-rpath "$rpath" "$f"
      fi
    done

    log "SDK installation complete"
  '';

  meta = {
    description = "MLIR + StableHLO + PJRT C API SDK derived from JAX->XLA lock.json";
    license = lib.licenses.asl20;
  };
}
