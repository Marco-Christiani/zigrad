# nix/xla-mlir-stablehlo-capi-sdk.nix
{
  lib,
  stdenv,
  fetchurl,
  runCommand,
  cmake,
  ninja,
  python3,
  perl,
  unzip,
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
  devel ? false,
}: let
  lock = builtins.fromJSON (builtins.readFile lockFile);
  inherit (lock) pins;
  inherit (pins) xla llvm stablehlo;

  xlaTar = fetchurl {
    url = xla.tarball_url;
    hash = xla.hash_sri;
  };
  llvmTar = fetchurl {
    inherit (llvm) urls;
    hash = llvm.hash_sri;
  };
  stablehloZip = fetchurl {
    inherit (stablehlo) urls;
    hash = stablehlo.hash_sri;
  };

  llvmPatches = ["build.patch" "mathextras.patch" "toolchains.patch" "zstd.patch" "lit_test.patch"];
  llvmIgnoredPatches = ["generated.patch"];

  stablehloPatches = ["temporary.patch"];

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

  stablehloSrc = runCommand "stablehlo-src-${builtins.substring 0 12 stablehlo.commit}" {nativeBuildInputs = [unzip patch perl];} ''
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
        *) echo "ERROR: New or unexpected StableHLO patch detected: $p" >&2; exit 1;;
      esac
    done
    for p in $expected_patches; do
      echo "[stablehlo] Applying $p"
      patch -p1 < "${xlaSrc}/third_party/stablehlo/$p"
    done

    # Force ONLY StablehloCAPI to be SHARED without turning on BUILD_SHARED_LIBS globally for StableHLO.
    # This relies on MLIR's add_mlir_public_c_api_library forwarding args into add_mlir_library which supports SHARED.
    if [ -f stablehlo/integrations/c/CMakeLists.txt ]; then
      perl -pi -e 's/add_mlir_public_c_api_library\(StablehloCAPI/add_mlir_public_c_api_library(StablehloCAPI SHARED/' \
        stablehlo/integrations/c/CMakeLists.txt
    fi

    # Sanity check
    grep -n "add_mlir_public_c_api_library(StablehloCAPI" -n stablehlo/integrations/c/CMakeLists.txt || true
    grep -n "StablehloCAPI SHARED" stablehlo/integrations/c/CMakeLists.txt || true

    echo "[stablehlo] post patch stablehlo/integrations/c/CMakeLists.txt:"
    cat stablehlo/integrations/c/CMakeLists.txt
  '';
in
  stdenv.mkDerivation {
    pname = "xla-mlir-stablehlo-capi-sdk";
    version = "xla-${builtins.substring 0 12 xla.commit}" + lib.optionalString devel "-devel";

    strictDeps = true;
    dontUnpack = true;
    dontConfigure = true;
    dontStrip = true;

    nativeBuildInputs = [
      cmake
      ninja
      python3
      perl
      patchelf
      lld
      binutils

      # host-tool runtime deps (strictDeps)
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

    buildPhase = ''
      set -euo pipefail

      # LLVM/MLIR: shared implementation libraries so MLIR-C and StablehloCAPI share one MLIR/LLVM instance at runtime.
      cmake_flags=(
        -G Ninja
        -DCMAKE_BUILD_TYPE=Release

        -DBUILD_SHARED_LIBS=ON
        -DLLVM_ENABLE_PROJECTS=mlir
        -DLLVM_TARGETS_TO_BUILD=host

        -DLLVM_INCLUDE_TESTS=OFF
        -DLLVM_INCLUDE_EXAMPLES=OFF
        -DLLVM_INCLUDE_DOCS=OFF
        -DMLIR_INCLUDE_TESTS=OFF
        -DMLIR_ENABLE_BINDINGS_PYTHON=OFF

        -DMLIR_BUILD_MLIR_C_DYLIB=ON

        # PICK ONE
        # Faster linker if available (check StableHLO docs).
        # -DLLVM_ENABLE_LLD=ON
        -DLLVM_USE_LINKER=lld

        # RPATH - need libstdc++ DSO
        -DCMAKE_BUILD_RPATH=${lib.makeLibraryPath [stdenv.cc.cc.lib zlib zstd]}
        # could also set CMAKE_INSTALL_RPATH to the sam val too, not sure yet, we patchelf later
        # -DCMAKE_INSTALL_RPATH=...
        -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON
      )

      mkdir -p llvm-build
      cmake -S ${llvmSrc}/llvm -B llvm-build "''${cmake_flags[@]}"

      # Minimum host tools typically needed by StableHLO generation.
      cmake --build llvm-build --target llvm-tblgen mlir-tblgen

      # Build MLIR-C first (your Zig links against this).
      cmake --build llvm-build --target MLIR-C

      # If StableHLO’s build links to MLIRCAPI*.so imported targets, they must exist before stablehlo-build runs.
      # Building them here avoids the “missing and no known rule to make it” failure.
      cmake --build llvm-build --target \
        MLIRCAPIIR \
        MLIRCAPIArith \
        MLIRCAPIMath \
        MLIRCAPISCF \
        MLIRCAPITransforms \
        MLIRCAPIFunc \
        MLIRCAPITensor

      mkdir -p stablehlo-build
      cmake -S ${stablehloSrc} -B stablehlo-build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=OFF \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DSTABLEHLO_ENABLE_BINDINGS_PYTHON=OFF \
        -DMLIR_DIR="$PWD/llvm-build/lib/cmake/mlir" \
        -DLLVM_DIR="$PWD/llvm-build/lib/cmake/llvm"

      ninja -C stablehlo-build StablehloCAPI
    '';

    installPhase = ''
      set -euo pipefail
      log() { echo "[xla-mlir-stablehlo-capi-sdk] $*" >&2; }

      mkdir -p "$out/include" "$out/lib"

      copy_headers() {
        local src="$1"
        local dst="$2"
        mkdir -p "$dst"
        cp -r "$src"/* "$dst/"
      }

      # --- Headers (same layout you already rely on) ---
      copy_headers "${llvmSrc}/mlir/include/mlir-c" "$out/include/mlir-c"

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

      copy_headers "${stablehloSrc}/stablehlo/integrations/c" "$out/include/stablehlo/integrations/c"

      mkdir -p "$out/include/xla/pjrt/c"
      cp -v "${xlaSrc}/xla/pjrt/c/"*.h "$out/include/xla/pjrt/c/"

      # --- Libraries ---
      if [ "${lib.boolToString devel}" = "true" ]; then
        log "Devel mode: copying ALL build artifacts (*.so*, *.a)"
        find llvm-build stablehlo-build -type f \( -name "*.a" -o -name "*.so*" \) -print -exec cp -v {} "$out/lib/" \;
      else
        log "Minimal mode: copying MLIR-C + StablehloCAPI and DT_NEEDED closure"

        # Copy the two "roots"
        cp -v llvm-build/lib/libMLIR-C.so* "$out/lib/" || true
        cp -v stablehlo-build/lib/libStablehloCAPI.so* "$out/lib/" || true

        # Copy their transitive deps from build trees into $out/lib (but do not try to vendor glibc/libstdc++ here).
        copy_needed_closure() {
          local search_dirs=("$PWD/llvm-build/lib" "$PWD/stablehlo-build/lib")
          local -A seen
          local queue=("$@")

          while [ "''${#queue[@]}" -gt 0 ]; do
            local libpath="''${queue[0]}"
            queue=("''${queue[@]:1}")

            [ -f "$libpath" ] || continue
            local base="$(basename "$libpath")"
            if [ -n "''${seen[$base]:-}" ]; then
              continue
            fi
            seen["$base"]=1

            # Ensure it's present in output
            if [ ! -f "$out/lib/$base" ]; then
              cp -v "$libpath" "$out/lib/"
            fi

            # Enqueue internal deps that we can find in search_dirs
            while IFS= read -r need; do
              local found=""
              for d in "''${search_dirs[@]}"; do
                if [ -f "$d/$need" ]; then
                  found="$d/$need"
                  break
                fi
              done
              if [ -n "$found" ]; then
                queue+=("$found")
              fi
            done < <(patchelf --print-needed "$libpath" || true)
          done
        }

        mlir_root="$(ls -1 llvm-build/lib/libMLIR-C.so.* 2>/dev/null | head -n1 || true)"
        stablehlo_root="$(ls -1 stablehlo-build/lib/libStablehloCAPI.so.* 2>/dev/null | head -n1 || true)"
        copy_needed_closure "$mlir_root" "$stablehlo_root"
      fi

      # Ensure unversioned linker names exist for the two link-entry DSOs
      for name in libMLIR-C libStablehloCAPI; do
        so="$(ls -1 "$out/lib/$name.so."* 2>/dev/null | head -n1 || true)"
        if [ -n "$so" ]; then
          ln -sfn "$(basename "$so")" "$out/lib/$name.so"
        fi
      done

      # Patch RUNPATH for *all* shipped DSOs (not just MLIR-C).
      rpath="\$ORIGIN:\$ORIGIN/../runtime/sys/lib:${
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

      # hard fail on critical ones
      for f in "$out/lib/"*.so*; do
        [ -f "$f" ] || continue
        if ! patchelf --set-rpath "$rpath" "$f"; then
          case "$(basename "$f")" in
            libMLIR-C.so*|libStablehloCAPI.so*)
              log "ERROR: patchelf failed for critical DSO: $f"
              exit 1
              ;;
            *)
              log "WARNING: patchelf failed for non-critical DSO: $f"
              ;;
          esac
        fi
      done

      # Sanity check... C API dialect handle function should be present when StablehloCAPI is built properly.
      if [ -f "$out/lib/libStablehloCAPI.so" ]; then
        if ! nm -D "$out/lib/libStablehloCAPI.so" 2>/dev/null | grep -q "mlirGetDialectHandle__stablehlo__"; then
          # TODO: maybe this should be a hard error?
          log "WARNING: mlirGetDialectHandle__stablehlo__ not found in libStablehloCAPI.so (symbol export may differ by version/config)"
        fi
      fi

      log "SDK installation complete"
    '';

    meta = {
      description = "MLIR + StableHLO + PJRT C API SDK derived from JAX->XLA lock.json";
      license = lib.licenses.asl20;
    };
  }
